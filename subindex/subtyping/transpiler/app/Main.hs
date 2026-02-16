module Main (main) where

import Control.Monad (void, when)
import Data.Char (isAlphaNum, isDigit, isSpace, toUpper)
import Data.List (isPrefixOf, isSuffixOf, stripPrefix)
import Data.Maybe (fromMaybe, isJust, isNothing)
import System.Environment (getArgs)
import System.Exit (exitFailure)
import Text.Parsec

-- Signature spec parsed from {T @ g :: c1 | c2 | ...}
data SigSpec = SigSpec
  { sigTarget :: String
  , sigDelta :: String
  , sigConstructors :: [String]
  }
  deriving (Show)

-- Branch in match ... with | C x => body ... end
data Branch = Branch
  { branchCtor :: String
  , branchVar :: String
  , branchBody :: String
  }
  deriving (Show)

-- Top-level statements of the surface language.
data Stmt
  = StmtDef String String String
  | StmtInductive IndInfo Bool
  | StmtRaw String
  deriving (Show)

data IndInfo = IndInfo
  { indName :: String
  , indCtors :: [String]
  , indIndexesExpr :: String
  , indPatternExpr :: String
  , indDomainKinds :: [DomainKind]
  }
  deriving (Show)

data DomainKind
  = DKPoly
  | DKNat
  | DKOther
  deriving (Show, Eq)

main :: IO ()
main = do
  args <- getArgs
  case args of
    [inputPath] -> do
      src <- readFile inputPath
      case transpileProgram src of
        Left err -> dieWith err
        Right out -> putStrLn out
    [inputPath, outputPath] -> do
      src <- readFile inputPath
      case transpileProgram src of
        Left err -> dieWith err
        Right out -> writeFile outputPath out
    _ -> dieWith "Usage: siglp-transpiler <input.siglp> [output.lp]"

transpileProgram :: String -> Either String String
transpileProgram src =
  case runParser programP [] "<siglp>" src of
    Left err -> Left (show err)
    Right stmts ->
      let rendered = unlines (filter (not . null) (map renderStmt stmts))
          withPrelude =
            if "require open subtyping.encode;" `isInText` rendered
              then rendered
              else unlines ["require open subtyping.encode;", "", rendered]
       in Right withPrelude

renderStmt :: Stmt -> String
renderStmt (StmtDef name ty tm) =
  unlines
    [ "symbol " ++ name ++ " : " ++ ty ++ ";"
    , "rule " ++ name ++ " ↪ " ++ tm ++ ";"
    ]
renderStmt (StmtInductive info isFirst) =
  if isFirst
    then
      unlines
        [ "symbol " ++ sigTargetSym (indName info) ++ " : Set;"
        , "rule " ++ sigTargetSym (indName info) ++ " ↪ " ++ indName info ++ "Label;"
        , "symbol " ++ sigIndexesSym (indName info) ++ " : indexes_type;"
        , "rule " ++ sigIndexesSym (indName info) ++ " ↪ " ++ indIndexesExpr info ++ ";"
        , "symbol " ++ sigPatternSym (indName info) ++ " : index_pattern_list;"
        , "rule " ++ sigPatternSym (indName info) ++ " ↪ " ++ indPatternExpr info ++ ";"
        , "symbol " ++ sigDeltaSym (indName info) ++ " : delta_tel;"
        , "rule " ++ sigDeltaSym (indName info) ++ " ↪ mk_delta_tel " ++ sigIndexesSym (indName info) ++ " " ++ sigPatternSym (indName info) ++ ";"
        , "symbol " ++ sigAllPhiSym (indName info) ++ " : constructor_list;"
        , "rule " ++ sigAllPhiSym (indName info) ++ " ↪ " ++ ctorListExpr (knownCtorLabels (map normalizeCtorName (indCtors info))) ++ ";"
        ]
    else ""
renderStmt (StmtRaw s) =
  let t = trim s
   in if null t then "" else t ++ ";"

-- Program parser --------------------------------------------------------------

type P = Parsec String [IndInfo]

programP :: P [Stmt]
programP = sc *> many (statementP <* sc) <* eof

statementP :: P Stmt
statementP = try inductiveStmtP <|> try defStmtP <|> rawStmtP

inductiveStmtP :: P Stmt
inductiveStmtP = try $ do
  _ <- keyword "Inductive" <|> keyword "inductive"
  sc1
  name <- identifier <?> "inductive name"
  rest <- rawUntil (char ';')
  _ <- char ';'
  let ctors = extractCtorNames rest
  known <- getState
  let tySpec = extractInductiveType rest
  let domains = extractTypeDomains tySpec
  let idxExpr = inferIndexesExpr known domains
  let patExpr = inferAnyPatternExpr (length domains)
  let domKinds = map (classifyDomainKind known) domains
  let info = IndInfo name ctors idxExpr patExpr domKinds
  let isFirst = isNothing (lookupInd known name)
  when isFirst (putState (info : known))
  pure (StmtInductive info isFirst)

defStmtP :: P Stmt
defStmtP = do
  keyword "Def"
  sc1
  name <- identifier <?> "definition name"
  sc
  _ <- char ':' <?> "':' after definition name"
  sc
  ty <- trim <$> exprUntil (try (string ":="))
  _ <- string ":="
  sc
  tm <- trim <$> exprUntil (char ';')
  _ <- char ';'
  let tm' = autoWrapConstructorResult ty tm
  pure (StmtDef name ty tm')

rawStmtP :: P Stmt
rawStmtP = do
  stmt <- trim <$> exprUntil (char ';')
  _ <- char ';'
  when (startsWithKeyword "symbol" stmt) $
    fail "Source language forbids `symbol`; use `Def` declarations."
  when (startsWithKeyword "rule" stmt) $
    fail "Source language forbids `rule`; use `Def` declarations."
  pure (StmtRaw stmt)

-- Expression parser/rewriter -------------------------------------------------

exprUntil :: P end -> P String
exprUntil end = concat <$> manyTill exprPiece (lookAhead (try end))

exprPiece :: P String
exprPiece =
  choice
    [ try matchExprP
    , illegalMatchExprP
    , try signatureExprP
    , delimitedExpr '(' ')'
    , delimitedExpr '[' ']'
    , delimitedExpr '{' '}'
    , commentChunk
    , whitespaceChunk
    , identChunk
    , singletonChunk
    ]

illegalMatchExprP :: P String
illegalMatchExprP = do
  keyword "match"
  fail "Only signature matches are supported: `match m as {T :: c1 | ...} return Q with ... end`."

matchExprP :: P String
matchExprP = try $ do
  known <- getState
  keyword "match"
  sc1
  scrut <- trim <$> exprUntil (keyword "as")
  keyword "as"
  sc
  sig <- between (char '{') (char '}') signatureSpecP
  sc
  keyword "return"
  sc1
  qTy <- trim <$> exprUntil (keyword "with")
  keyword "with"
  sc
  branches <- many1 branchP
  keyword "end"

  when (not (isBackendSignature sig)) $
    fail "Match `as { ... }` must denote a supported signature (constructor labels known by backend)."

  let qTySet = normalizeMatchReturnQ known qTy
  ordered <- either fail pure (orderBranches (sigConstructors sig) branches)
  let phiExpr = ctorListExpr (sigConstructors sig)
  let branchExpr = buildCaseBranches qTySet ordered

  pure
    ( "(signature_case "
        ++ sigTarget sig
        ++ " "
        ++ sigDelta sig
        ++ " ("
        ++ phiExpr
        ++ ") "
        ++ qTySet
        ++ " "
        ++ scrut
        ++ " ("
        ++ branchExpr
        ++ "))"
    )

branchP :: P Branch
branchP = do
  _ <- char '|'
  sc
  ctorRaw <- identifier <?> "constructor label"
  let ctor = normalizeCtorName ctorRaw
  sc
  mVar <- optionMaybe identifier
  sc
  _ <- string "=>" <?> "'=>' in branch"
  sc
  body <- trim <$> exprUntil branchEndP
  sc
  pure (Branch ctor (fromMaybe "k" mVar) body)

branchEndP :: P ()
branchEndP =
  choice
    [ void (char '|')
    , void (keyword "end")
    ]

signatureExprP :: P String
signatureExprP = try $ do
  sig <- between (char '{') (char '}') signatureSpecP
  if isBackendSignature sig
    then
      pure
        ( "(signature_term "
            ++ sigTarget sig
            ++ " "
            ++ sigDelta sig
            ++ " ("
            ++ ctorListExpr (sigConstructors sig)
            ++ "))"
        )
    else
      case natSubsetFallback sig of
        Just t -> pure t
        Nothing ->
          fail
            ( "Unsupported signature expression constructors: "
                ++ show (sigConstructors sig)
                ++ "."
            )

signatureSpecP :: P SigSpec
signatureSpecP = do
  left <- trim <$> rawUntil (choice [void (try (string "::")), void (lookAhead (char '|')), void (lookAhead (char '}'))])
  known <- getState
  (t, g, defaults) <- either fail pure (resolveTargetDelta known left)

  explicitCtorList <-
    optionMaybe
      ( do
          _ <- lookAhead (try (string "::") <|> string "|")
          _ <- optional (try (string "::"))
          _ <- optional (char '|')
          sc
          firstCtor <- trim <$> rawUntil (choice [void (char '|'), void (lookAhead (char '}'))])
          more <-
            many
              (char '|' *> (trim <$> rawUntil (choice [void (char '|'), void (lookAhead (char '}'))])))
          pure (filter (not . null) (map normalizeCtorName (firstCtor : more)))
      )

  let ctors = fromMaybe defaults explicitCtorList
  when (isJust explicitCtorList) $
    case runParser targetDeltaP [] "<signature-left>" left of
      Right (TDImplicitName n) ->
        case lookupInd known n of
          Just info ->
            when (not (null (indDomainKinds info))) $
              fail
                ( "Signature filter for `" ++ n
                    ++ "` requires explicit telescope indices. Use `{"
                    ++ n
                    ++ " ... | c1 | ...}`."
                )
          Nothing -> pure ()
      _ -> pure ()

  when (null ctors) $
    fail "Signature must provide constructors via `::`, or use a declared inductive with known constructors."
  pure (SigSpec t g ctors)

rawUntil :: P end -> P String
rawUntil end = concat <$> manyTill rawPiece (lookAhead (try end))

rawPiece :: P String
rawPiece =
  choice
    [ delimitedRaw '(' ')'
    , delimitedRaw '[' ']'
    , delimitedRaw '{' '}'
    , commentChunk
    , whitespaceChunk
    , identChunk
    , singletonChunk
    ]

delimitedExpr :: Char -> Char -> P String
delimitedExpr open close = do
  _ <- char open
  inner <- exprUntil (char close)
  _ <- char close
  pure (open : inner ++ [close])

delimitedRaw :: Char -> Char -> P String
delimitedRaw open close = do
  _ <- char open
  inner <- rawUntil (char close)
  _ <- char close
  pure (open : inner ++ [close])

commentChunk :: P String
commentChunk = try $ do
  _ <- string "//"
  body <- manyTill anyChar (lookAhead (void newline <|> eof))
  eol <- optionMaybe newline
  pure ("//" ++ body ++ maybe "" (const "\n") eol)

whitespaceChunk :: P String
whitespaceChunk = many1 (satisfy isSpace)

identChunk :: P String
identChunk = many1 (satisfy isIdentChar)

singletonChunk :: P String
singletonChunk = (: []) <$> anyChar

-- Helpers --------------------------------------------------------------------

data SigTermInfo = SigTermInfo
  { stiTarget :: String
  , stiDelta :: String
  , stiPhiExpr :: String
  , stiPhiList :: [String]
  }

autoWrapConstructorResult :: String -> String -> String
autoWrapConstructorResult ty tm =
  case parseSignatureTermInfo (resultTypeOfType ty) of
    Nothing -> tm
    Just info -> rewriteLambdaResult info tm

resultTypeOfType :: String -> String
resultTypeOfType ty = go (trim ty)
  where
    go t =
      let u = stripOuterParens (trim t)
       in case splitPiBody u of
            Just body -> go body
            Nothing ->
              let parts = filter (not . null) (map trim (splitTopLevelArrows u))
               in if null parts then u else last parts

splitPiBody :: String -> Maybe String
splitPiBody t
  | startsWithPi t =
      case findTopLevelChar ',' t of
        Just i -> Just (trim (drop (i + 1) t))
        Nothing -> Nothing
  | otherwise = Nothing

startsWithPi :: String -> Bool
startsWithPi t = "Π" `isPrefixOf` t || "forall" `isPrefixOf` t

parseSignatureTermInfo :: String -> Maybe SigTermInfo
parseSignatureTermInfo s = do
  let u = stripOuterParens (trim s)
  chunks <- case splitTopLevelWords u of
    ("signature_term" : xs) | length xs >= 3 -> Just xs
    _ -> Nothing
  let target = chunks !! 0
  let delta = chunks !! 1
  let phiExpr = trim (unwords (drop 2 chunks))
  phi <- parseCtorListExpr phiExpr
  pure (SigTermInfo target delta phiExpr phi)

parseCtorListExpr :: String -> Maybe [String]
parseCtorListExpr raw =
  let u = stripOuterParens (trim raw)
      ws = splitTopLevelWords u
   in case ws of
        ["constructor_list_nil"] -> Just []
        ("constructor_list_cons" : c : rest) -> do
          let tailExpr = trim (unwords rest)
          cs <- parseCtorListExpr tailExpr
          pure (normalizeCtorName c : cs)
        _ -> Nothing

rewriteLambdaResult :: SigTermInfo -> String -> String
rewriteLambdaResult info tm =
  case splitLambdaPrefix tm of
    Just (prefix, body) -> prefix ++ ", " ++ rewriteLambdaResult info body
    Nothing -> wrapConstructorLeaf info tm

splitLambdaPrefix :: String -> Maybe (String, String)
splitLambdaPrefix raw =
  let t = trim raw
   in if isLambdaExpr t
        then
          case findTopLevelChar ',' t of
            Just i ->
              let prefix = trim (take i t)
                  body = trim (drop (i + 1) t)
               in if null body then Nothing else Just (prefix, body)
            Nothing -> Nothing
        else Nothing

isLambdaExpr :: String -> Bool
isLambdaExpr [] = False
isLambdaExpr (c : _) = c == 'λ' || c == '\\'

wrapConstructorLeaf :: SigTermInfo -> String -> String
wrapConstructorLeaf info rawLeaf =
  let leaf = trim rawLeaf
      headLabel = constructorHeadLabel leaf
   in if startsWithKeyword "signature_intro" leaf
        then rawLeaf
        else
          case headLabel of
            Just c | c `elem` stiPhiList info ->
              "(signature_intro "
                ++ stiTarget info
                ++ " "
                ++ stiDelta info
                ++ " "
                ++ stiPhiExpr info
                ++ " "
                ++ c
                ++ " top "
                ++ leaf
                ++ ")"
            _ -> rawLeaf

constructorHeadLabel :: String -> Maybe String
constructorHeadLabel raw =
  let t = stripOuterParens (trim raw)
   in case splitTopLevelWords t of
        (h : _) ->
          let lbl = normalizeCtorName h
           in if lbl `elem` backendCtorLabels then Just lbl else Nothing
        _ -> Nothing

findTopLevelChar :: Char -> String -> Maybe Int
findTopLevelChar needle = go 0 0 0 0
  where
    go _ _ _ _ [] = Nothing
    go p b c i (x : xs)
      | x == '(' = go (p + 1) b c (i + 1) xs
      | x == ')' = go (max 0 (p - 1)) b c (i + 1) xs
      | x == '[' = go p (b + 1) c (i + 1) xs
      | x == ']' = go p (max 0 (b - 1)) c (i + 1) xs
      | x == '{' = go p b (c + 1) (i + 1) xs
      | x == '}' = go p b (max 0 (c - 1)) (i + 1) xs
      | x == needle && p == 0 && b == 0 && c == 0 = Just i
      | otherwise = go p b c (i + 1) xs

data TargetDeltaLeft
  = TDExplicitAt String String
  | TDSpace String String
  | TDImplicitName String

resolveTargetDelta :: [IndInfo] -> String -> Either String (String, String, [String])
resolveTargetDelta known txt =
  case runParser targetDeltaP [] "<signature-left>" txt of
    Left err -> Left (show err)
    Right (TDExplicitAt t g) -> Right (t, g, [])
    Right (TDSpace t g) ->
      case lookupInd known t of
        Just info ->
          let args = splitTopLevelWords g
           in if length args == length (indDomainKinds info)
                then do
                  pat <- buildPatternFromArgs (indDomainKinds info) args
                  let delta = "(mk_delta_tel " ++ sigIndexesSym t ++ " (" ++ pat ++ "))"
                  pure (sigTargetSym t, delta, knownCtorLabels (map normalizeCtorName (indCtors info)))
                else Right (t, g, [])
        Nothing -> Right (t, g, [])
    Right (TDImplicitName n) ->
      case lookupInd known n of
        Just info -> Right (sigTargetSym n, sigDeltaSym n, knownCtorLabels (map normalizeCtorName (indCtors info)))
        Nothing -> Left ("Unknown inductive `" ++ n ++ "` in signature. Declare `Inductive " ++ n ++ " ...;` first, or use `{T @ g :: ...}`.")

-- Left side of signature: either `T @ g` (general), or `T g` (simple fallback).
targetDeltaP :: P TargetDeltaLeft
targetDeltaP = try atForm <|> try spaceForm <|> nameOnly
  where
    atForm = do
      t <- trim <$> rawUntil (char '@')
      _ <- char '@'
      g <- (trim . concat) <$> many rawPiece
      eof
      when (null t || null g) $ fail "Malformed '{T @ g :: ...}' left side."
      pure (TDExplicitAt t g)

    -- Backward compatibility: {T g :: ...}
    spaceForm = do
      t <- identifier
      sc1
      g <- (trim . concat) <$> many rawPiece
      eof
      when (null g) $ fail "Malformed '{T g :: ...}' left side."
      pure (TDSpace t g)

    -- New shorthand: {vec :: ...} where vec was declared by `Inductive vec ...;`
    nameOnly = do
      n <- identifier
      sc
      eof
      pure (TDImplicitName n)

orderBranches :: [String] -> [Branch] -> Either String [Branch]
orderBranches ctors bs = do
  let duplicates = findDuplicates (map branchCtor bs)
  if not (null duplicates)
    then Left ("Duplicate match branch for constructor: " ++ head duplicates)
    else do
      ordered <- mapM pick ctors
      let extras = [c | c <- map branchCtor bs, c `notElem` ctors]
      if not (null extras)
        then Left ("Branch constructor not in signature: " ++ head extras)
        else pure ordered
  where
    pick c =
      case [b | b <- bs, branchCtor b == c] of
        [] -> Left ("Missing branch for constructor: " ++ c)
        (b : _) -> Right b

buildCaseBranches :: String -> [Branch] -> String
buildCaseBranches qTy branches = go branches
  where
    go [] = "case_branches_nil " ++ qTy
    go (b : bs) =
      let tailCtors = map branchCtor bs
          lam =
            "(λ ("
              ++ branchVar b
              ++ " : constructor_type "
              ++ branchCtor b
              ++ "), "
              ++ branchBody b
              ++ ")"
       in "case_branches_cons "
            ++ branchCtor b
            ++ " ("
            ++ ctorListExpr tailCtors
            ++ ") "
            ++ qTy
            ++ " "
            ++ lam
            ++ " ("
            ++ go bs
            ++ ")"

ctorListExpr :: [String] -> String
ctorListExpr [] = "constructor_list_nil"
ctorListExpr (c : cs) = "constructor_list_cons " ++ c ++ " (" ++ ctorListExpr cs ++ ")"

keyword :: String -> P ()
keyword kw = try (string kw *> notFollowedBy (satisfy isIdentChar))

identifier :: P String
identifier = do
  c <- satisfy isIdentStart
  cs <- many (satisfy isIdentChar)
  pure (c : cs)

sc :: P ()
sc = skipMany sepUnit

sc1 :: P ()
sc1 = skipMany1 sepUnit

sepUnit :: P ()
sepUnit = void (satisfy isSpace) <|> void commentChunk

isIdentStart :: Char -> Bool
isIdentStart c = c == '_' || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')

isIdentChar :: Char -> Bool
isIdentChar c = isAlphaNum c || c == '_' || c == '\''

startsWithKeyword :: String -> String -> Bool
startsWithKeyword kw s =
  let t = dropWhile isSpace s
   in kw `isPrefixOf` t
        && let rest = drop (length kw) t
            in null rest || not (isIdentChar (head rest))

normalizeCtorName :: String -> String
normalizeCtorName c =
  if "Label" `isSuffixOf` c
    then c
    else c ++ "Label"

lookupInd :: [IndInfo] -> String -> Maybe IndInfo
lookupInd env n =
  case [i | i <- env, indName i == n] of
    [] -> Nothing
    (i : _) -> Just i

extractCtorNames :: String -> [String]
extractCtorNames s =
  [ name
  | part <- drop 1 (splitOnChar '|' s)
  , let name = takeWhile isIdentChar (dropWhile (\c -> isSpace c || c == '|') part)
  , not (null name)
  ]

splitOnChar :: Char -> String -> [String]
splitOnChar sep = go []
  where
    go acc [] = [reverse acc]
    go acc (c : cs)
      | c == sep = reverse acc : go [] cs
      | otherwise = go (c : acc) cs

extractInductiveType :: String -> String
extractInductiveType rest =
  let afterColon =
        case dropWhile (/= ':') rest of
          ':' : xs -> xs
          _ -> rest
   in trim (takeBeforeAny ["≔", ":=", " where "] afterColon)

takeBeforeAny :: [String] -> String -> String
takeBeforeAny needles s =
  case foldr pick Nothing needles of
    Nothing -> s
    Just i -> take i s
  where
    pick needle acc =
      case findSubstr needle s of
        Nothing -> acc
        Just i ->
          case acc of
            Nothing -> Just i
            Just j -> Just (min i j)

findSubstr :: String -> String -> Maybe Int
findSubstr needle hay = go 0 hay
  where
    go _ [] = if null needle then Just 0 else Nothing
    go i xs@(_ : ys)
      | needle `isPrefixOf` xs = Just i
      | otherwise = go (i + 1) ys

extractTypeDomains :: String -> [String]
extractTypeDomains ty =
  let parts = filter (not . null) (map trim (splitTopLevelArrows ty))
   in case reverse parts of
        (r : rs) | looksType r -> reverse rs
        _ -> []

splitTopLevelArrows :: String -> [String]
splitTopLevelArrows = go 0 0 0 []
  where
    go _ _ _ cur [] = [trim (reverse cur)]
    go p b c cur ('-' : '>' : xs)
      | p == 0 && b == 0 && c == 0 = trim (reverse cur) : go p b c [] xs
    go p b c cur ('→' : xs)
      | p == 0 && b == 0 && c == 0 = trim (reverse cur) : go p b c [] xs
    go p b c cur (x : xs)
      | x == '(' = go (p + 1) b c (x : cur) xs
      | x == ')' = go (max 0 (p - 1)) b c (x : cur) xs
      | x == '[' = go p (b + 1) c (x : cur) xs
      | x == ']' = go p (max 0 (b - 1)) c (x : cur) xs
      | x == '{' = go p b (c + 1) (x : cur) xs
      | x == '}' = go p b (max 0 (c - 1)) (x : cur) xs
      | otherwise = go p b c (x : cur) xs

looksType :: String -> Bool
looksType s =
  let u = map toUpper (filter (not . isSpace) s)
   in u == "TYPE"

inferIndexesExpr :: [IndInfo] -> [String] -> String
inferIndexesExpr env domains = foldr step "index_null" (map (inferIndexStep env) domains)
  where
    step (Left ()) acc = "index_poly_succ (" ++ acc ++ ")"
    step (Right lbl) acc = "index_dependent_lift_succ " ++ lbl ++ " (" ++ acc ++ ")"

classifyDomainKind :: [IndInfo] -> String -> DomainKind
classifyDomainKind env dom =
  case inferIndexStep env dom of
    Left () -> DKPoly
    Right "natLabel" -> DKNat
    Right _ -> DKOther

inferIndexStep :: [IndInfo] -> String -> Either () String
inferIndexStep env dom =
  let t = normalizeDomainTarget (stripBinderType (trim dom))
   in if t == "Set"
        then Left ()
        else
          case stripPrefix "lift " t of
            Just lbl -> Right (trimParens (trim lbl))
            Nothing ->
              if t == "nat" || t == "Nat"
                then Right "natLabel"
                else
                  case lookupInd env t of
                    Just _ -> Right (t ++ "Label")
                    Nothing ->
                      if "Label" `isSuffixOf` t
                        then Right t
                        else Left ()

normalizeDomainTarget :: String -> String
normalizeDomainTarget t =
  case parseSignatureLikeDomain t of
    Just (baseTy, _) -> baseTy
    Nothing -> t

-- Parse domain forms used in inductive binders, e.g.:
--   {nat | 0 | 1}
--   {nat :: _0 | +1}
-- We currently use only the base target (`nat` above) to infer index kind.
parseSignatureLikeDomain :: String -> Maybe (String, [String])
parseSignatureLikeDomain raw =
  let t = trim raw
   in if length t >= 2 && head t == '{' && last t == '}'
        then
          let inner = trim (init (tail t))
              (lhs, rhs) =
                case findSubstr "::" inner of
                  Just i -> (trim (take i inner), drop (i + 2) inner)
                  Nothing ->
                    case findSubstr "|" inner of
                      Just i -> (trim (take i inner), drop (i + 1) inner)
                      Nothing -> (trim inner, "")
              ctors = filter (not . null) (map trim (splitOnChar '|' rhs))
           in if null lhs then Nothing else Just (lhs, ctors)
        else Nothing

buildPatternFromArgs :: [DomainKind] -> [String] -> Either String String
buildPatternFromArgs kinds args
  | length kinds /= length args =
      Left "Indexed signature application has wrong number of arguments for the inductive."
  | otherwise = do
      atoms <- sequence (zipWith argToAtom kinds args)
      pure (foldr (\a acc -> "index_pattern_cons (" ++ a ++ ") (" ++ acc ++ ")") "index_pattern_nil" atoms)
  where
    argToAtom DKPoly _ = Right "index_pat_var"
    argToAtom DKOther _ = Right "index_pat_var"
    argToAtom DKNat arg =
      case parseNatPatternTerm arg of
        Nothing -> Left ("Unsupported nat index pattern: `" ++ trim arg ++ "`.")
        Just "index_term_var" -> Right "index_pat_var"
        Just t -> Right ("index_pat_term (" ++ t ++ ")")

parseNatPatternTerm :: String -> Maybe String
parseNatPatternTerm raw =
  let t0 = trim raw
      t = stripOuterParens t0
   in if t == "_" || isSimpleVar t
        then Just "index_term_var"
        else
          if t == "_0" || t == "0"
            then Just "index_term_ctor _0 index_term_nil"
            else
              case
                  case stripPrefix "+1" t of
                    Just r -> Just r
                    Nothing -> stripPrefix "S" t
                of
                Just rest ->
                  let r = trim rest
                   in if null r
                        then Nothing
                        else do
                          sub <- parseNatPatternTerm r
                          Just ("index_term_ctor (+1 _0) (index_term_cons " ++ sub ++ " index_term_nil)")
                Nothing -> Nothing

isSimpleVar :: String -> Bool
isSimpleVar [] = False
isSimpleVar (c : cs) = isIdentStart c && all isIdentChar cs

stripOuterParens :: String -> String
stripOuterParens s =
  let t = trim s
   in if hasBalancedOuterParens t then stripOuterParens (trim (init (tail t))) else t

hasBalancedOuterParens :: String -> Bool
hasBalancedOuterParens t =
  not (null t)
    && head t == '('
    && last t == ')'
    && enclosesAll t
  where
    enclosesAll = go 0 0
    go _ _ [] = True
    go d i (x : xs)
      | x == '(' = go (d + 1) (i + 1) xs
      | x == ')' =
          let d' = d - 1
           in d' >= 0
                && (if d' == 0 then i == length t - 1 else True)
                && go d' (i + 1) xs
      | otherwise = go d (i + 1) xs

splitTopLevelWords :: String -> [String]
splitTopLevelWords = filter (not . null) . map trim . go 0 0 0 []
  where
    go _ _ _ cur [] = [reverse cur]
    go p b c cur (x : xs)
      | isSpace x && p == 0 && b == 0 && c == 0 = reverse cur : go p b c [] xs
      | x == '(' = go (p + 1) b c (x : cur) xs
      | x == ')' = go (max 0 (p - 1)) b c (x : cur) xs
      | x == '[' = go p (b + 1) c (x : cur) xs
      | x == ']' = go p (max 0 (b - 1)) c (x : cur) xs
      | x == '{' = go p b (c + 1) (x : cur) xs
      | x == '}' = go p b (max 0 (c - 1)) (x : cur) xs
      | otherwise = go p b c (x : cur) xs

stripBinderType :: String -> String
stripBinderType t =
  if not (null t) && (head t == '(' || head t == '[') && (last t == ')' || last t == ']')
    then
      let inner = trim (tail (init t))
       in if ':' `elem` inner
            then trim (tail (dropWhile (/= ':') inner))
            else t
    else t

trimParens :: String -> String
trimParens t =
  if not (null t) && head t == '(' && last t == ')'
    then trim (init (tail t))
    else t

inferAnyPatternExpr :: Int -> String
inferAnyPatternExpr n = foldr (\_ acc -> "index_pattern_cons index_pat_var (" ++ acc ++ ")") "index_pattern_nil" [1 .. n]

knownCtorLabels :: [String] -> [String]
knownCtorLabels = filter (`elem` ["emptyLabel", "consLabel"])

backendCtorLabels :: [String]
backendCtorLabels = ["emptyLabel", "consLabel"]

isBackendSignature :: SigSpec -> Bool
isBackendSignature sig = all (`elem` backendCtorLabels) (sigConstructors sig)

natSubsetFallback :: SigSpec -> Maybe String
natSubsetFallback sig =
  case stripSigTargetName (sigTarget sig) of
    Just "nat" | all isNatSubsetLabel (sigConstructors sig) -> Just "nat"
    _ -> Nothing

stripSigTargetName :: String -> Maybe String
stripSigTargetName s =
  let suf = "__sig_target"
   in if suf `isSuffixOf` s
        then Just (take (length s - length suf) s)
        else Nothing

isNatSubsetLabel :: String -> Bool
isNatSubsetLabel lbl =
  lbl `elem` ["_0Label", "0Label", "+1Label", "1Label"]
    || case stripSuffixLabel lbl of
      Just n -> not (null n) && all isDigit n
      Nothing -> False

stripSuffixLabel :: String -> Maybe String
stripSuffixLabel s =
  let suf = "Label"
   in if suf `isSuffixOf` s
        then Just (take (length s - length suf) s)
        else Nothing

normalizeMatchReturnQ :: [IndInfo] -> String -> String
normalizeMatchReturnQ env qRaw =
  let q = stripOuterParens (trim qRaw)
   in if q == "nat" || q == "Nat"
        then "natLabel"
        else
          if isAlreadySetCode q
        then q
        else
          case lookupInd env q of
            Just _ -> q ++ "Label"
            Nothing -> q

isAlreadySetCode :: String -> Bool
isAlreadySetCode q =
  "Label" `isSuffixOf` q
    || "__sig_target" `isSuffixOf` q
    || "label " `isPrefixOf` q

sigTargetSym :: String -> String
sigTargetSym n = n ++ "__sig_target"

sigIndexesSym :: String -> String
sigIndexesSym n = n ++ "__sig_indexes"

sigPatternSym :: String -> String
sigPatternSym n = n ++ "__sig_pattern_any"

sigDeltaSym :: String -> String
sigDeltaSym n = n ++ "__sig_delta"

sigAllPhiSym :: String -> String
sigAllPhiSym n = n ++ "__sig_phi_all"

trim :: String -> String
trim = rtrim . ltrim
  where
    ltrim = dropWhile isSpace
    rtrim = reverse . dropWhile isSpace . reverse

isInText :: String -> String -> Bool
isInText needle hay = any (needle `isPrefixOf`) (tails hay)
  where
    tails [] = [[]]
    tails xs@(_ : xs') = xs : tails xs'

findDuplicates :: [String] -> [String]
findDuplicates = go [] []
  where
    go _ dups [] = reverse dups
    go seen dups (x : xs)
      | x `elem` seen && x `notElem` dups = go seen (x : dups) xs
      | otherwise = go (x : seen) dups xs

dieWith :: String -> IO a
dieWith msg = do
  putStrLn ("error: " ++ msg)
  exitFailure
