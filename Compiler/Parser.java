package parser;

public class Parser {
  // Recursive descent parser that inputs a C++Lite program and
  // generates its abstract syntax. Each method corresponds to
  // a concrete syntax grammar rule, which appears as a comment
  // at the beginning of the method.

  Token token; // current token from the input stream
  Lexer lexer;
  
  // 새로운 파서를 만든다.
  public Parser(Lexer ts) { // Open the C++Lite source program
    lexer = ts; // as a token stream, and
    token = lexer.next(); // retrieve its first Token
  }
  
  // 현재 토큰이 주어진 토큰 종류인지 확인하고 맞으면 다음 토큰으로 옮기고, 아니면 에러를 낸다.
  private String match(TokenType t) {
    String value = token.value();
    if (token.type().equals(t))
      token = lexer.next();
    else
      error(t);
    return value;
  }
  
  // 에러 출력
  private void error(TokenType tok) {
    System.err.println("Syntax error: expecting: " + tok + "; saw: " + token);
    System.exit(1);
  }
  // 에러 출력
  private void error(String tok) {
    System.err.println("Syntax error: expecting: " + tok + "; saw: " + token);
    System.exit(1);
  }
  
  //Program --> void main ( ) '{' Declarations Statements '}'
  public Program program() {
    TokenType[] header = {TokenType.Int, TokenType.Main, TokenType.LeftParen, TokenType.RightParen};
    for (int i = 0; i < header.length; i++) // bypass "int main ( )"
      match(header[i]);
    match(TokenType.LeftBrace);
    // System.out.print("call to declarations()");
    Declarations decs = declarations();
    Block b = progstatements();
    // System.out.println("block object call to statements()" + b);
    match(TokenType.RightBrace);
    // System.out.println(" Program() finished");
    return new Program(decs, b); // student exercise
  }

  //Declarations --> { Declaration }
  private Declarations declarations() {
    Declarations ds = new Declarations();
    
    // 더 이상의 타입이 없을 때까지 간다
    while (isType()) {
      declaration(ds);
    }
    // System.out.println(" arrary declared");
    return ds; // student exercise
  }
  
  //Declaration --> Type Identifier { , Identifier } ;
  private void declaration(Declarations ds) {
    Variable v;
    Declaration d;
    Type t = type();
    v = new Variable(match(TokenType.Identifier));
    d = new Declaration(v, t);
    ds.add(d);

    while (isComma()) {
      token = lexer.next();
      v = new Variable(match(TokenType.Identifier));
      d = new Declaration(v, t);
      // d = (v, t);
      ds.add(d);
    }
    match(TokenType.Semicolon);
  }

  // Type --> int | bool | float | char
  // look up enum in API amke sure that this is working
  private Type type() {
    Type t = null;
    if (token.type().equals(TokenType.Int)) {
      t = Type.INT;
      // System.out.println(" Type is int");
    } else if (token.type().equals(TokenType.Bool)) {
      t = Type.BOOL;
      // System.out.println(" Type is bool");
    } else if (token.type().equals(TokenType.Float)) {
      t = Type.FLOAT;
      // System.out.println(" Type is float");
    } else if (token.type().equals(TokenType.Char)) {
      t = Type.CHAR;
      // System.out.println(" Type is char");
    } else
      error("Error in Type construction");
    token = lexer.next();
    return t;
  }
  
  // Statement --> ; | Block | Assignment | IfStatement | WhileStatement
  private Statement statement() {
    Statement s = null;
    // System.out.println("starting statment()");
    if (token.type().equals(TokenType.Semicolon))
      s = new Skip();
    else if (token.type().equals(TokenType.LeftBrace)) // block
      s = statements();
    // System.out.println("block data " + s);}
    else if (token.type().equals(TokenType.If)) // if
      s = ifStatement();
    else if (token.type().equals(TokenType.While)) // while
      s = whileStatement();
    else if (token.type().equals(TokenType.Identifier))
      s = assignment();
    else
      error("Error in Statement construction");
    return s;
  }

  // Block --> '{' Statements '}'
  // System.out.println("Starting block statments() " );
  private Block statements() {
    Statement s;
    Block b = new Block();

    match(TokenType.LeftBrace);
    // System.out.println(" left brace matched");
    while (isStatement()) {
      s = statement();
      b.members.add(s);
    }
    match(TokenType.RightBrace);// end of the block
    return b;
  }

  // Block --> '{' Statements '}'
  // System.out.println("Starting block statments() " );
  // match(TokenType.LeftBrace);
  // System.out.println(" left brace matched");
  private Block progstatements() {
    Block b = new Block();
    Statement s;
    while (isStatement()) {
      s = statement();
      b.members.add(s);
    }
    // match(TokenType.RightBrace);// end of the block
    return b;
  }

  // Assignment --> Identifier = Expression ;
  // System.out.println("Starting assignment()");
  private Assignment assignment() {
    Expression source;
    Variable target;

    target = new Variable(match(TokenType.Identifier));
    match(TokenType.Assign);
    source = expression();
    match(TokenType.Semicolon);
    return new Assignment(target, source);
  }

  // IfStatement --> if ( Expression ) Statement [ else Statement ]
  private Conditional ifStatement() {
    Conditional con;
    Statement s;
    Expression test;

    match(TokenType.If);
    match(TokenType.LeftParen);
    test = expression();
    match(TokenType.RightParen);
    s = statement();
    if (token.type().equals(TokenType.Else)) {
      match(TokenType.Else);
      Statement elsestate = statement();
      con = new Conditional(test, s, elsestate);
    } else {
      con = new Conditional(test, s);
    }
    return con;
  }

  // WhileStatement --> while ( Expression ) Statement
  private Loop whileStatement() {
    Statement body;
    Expression test;

    match(TokenType.While);
    match(TokenType.LeftParen);
    test = expression();
    match(TokenType.RightParen);
    body = statement();
    return new Loop(test, body);

  }

  // Expression --> Conjunction { || Conjunction }
  // System.out.println("expression() start");
  private Expression expression() {
    Expression c = conjunction();
    while (token.type().equals(TokenType.Or)) {
      Operator op = new Operator(match(token.type()));
      Expression e = expression();
      c = new Binary(op, c, e);
    }
    return c; // student exercise
  }

  // Conjunction --> Equality { && Equality }
  // System.out.println("coonjunction() start");
  private Expression conjunction() {
    Expression eq = equality();
    while (token.type().equals(TokenType.And)) {
      Operator op = new Operator(match(token.type()));
      Expression c = conjunction();
      eq = new Binary(op, eq, c);
    }
    return eq;
  }

  // Equality --> Relation [ EquOp Relation ]
  // System.out.println("equality() start");
  private Expression equality() {
    Expression rel = relation();
    while (isEqualityOp()) {
      Operator op = new Operator(match(token.type()));
      Expression rel2 = relation();
      rel = new Binary(op, rel, rel2);
    }
    return rel; // student exercise
  }

  // Relation --> Addition [RelOp Addition]
  // System.out.println("relation() start");
  private Expression relation() {
    Expression a = addition();
    while (isRelationalOp()) {
      Operator op = new Operator(match(token.type()));
      Expression a2 = addition();
      a = new Binary(op, a, a2);
    }
    return a; // student exercise
  }

  // Addition --> Term { AddOp Term }
  // System.out.println("addition() start");
  private Expression addition() {
    Expression e = term();
    while (isAddOp()) {
      Operator op = new Operator(match(token.type()));
      Expression term2 = term();
      e = new Binary(op, e, term2);
    }
    return e;
  }

  // Term --> Factor { MultiplyOp Factor }
  // System.out.println("term() start");
  private Expression term() {
    Expression e = factor();
    while (isMultiplyOp()) {
      Operator op = new Operator(match(token.type()));
      Expression term2 = factor();
      e = new Binary(op, e, term2);
    }
    return e;
  }

  // Factor --> [ UnaryOp ] Primary
  // System.out.println("factor() start");
  private Expression factor() {
    if (isUnaryOp()) {
      Operator op = new Operator(match(token.type()));
      Expression term = primary();
      return new Unary(op, term);
    } else
      return primary();
  }

  // Primary --> Identifier | Literal | ( Expression )
  // | Type ( Expression )
  // System.out.println("(primary) start");
  private Expression primary() {
    Expression e = null;
    if (token.type().equals(TokenType.Identifier)) {
      e = new Variable(match(TokenType.Identifier));
    } else if (isLiteral()) {
      e = literal();
    } else if (token.type().equals(TokenType.LeftParen)) {
      token = lexer.next();
      e = expression();
      match(TokenType.RightParen);
    } else if (isType()) {
      Operator op = new Operator(match(token.type()));
      match(TokenType.LeftParen);
      Expression term = expression();
      match(TokenType.RightParen);
      e = new Unary(op, term);
    } else
      error("Identifier | Literal | ( | Type");
    return e;
  }

  private Value literal() { // take the stringy part and convert it to the correct return new typed
                            // value. cast it to the correct value
    Value value = null;
    String stval = token.value();
    // int leteral 일 때
    if (token.type().equals(TokenType.IntLiteral)) {
      value = new IntValue(Integer.parseInt(stval));
      token = lexer.next();
      // System.out.println("found int literal");
    }
    // float literal 일 때
    else if (token.type().equals(TokenType.FloatLiteral)) {
      value = new FloatValue(Float.parseFloat(stval));
      token = lexer.next();
    }
    // char literal 일 때
    else if (token.type().equals(TokenType.CharLiteral)) {
      value = new CharValue(stval.charAt(0));
      token = lexer.next();
    }
    // bool : True literal 일 때
    else if (token.type().equals(TokenType.True)) {
      value = new BoolValue(true);
      token = lexer.next();
    }
    // bool : False : litearl 일 때
    else if (token.type().equals(TokenType.False)) {
      value = new BoolValue(false);
      token = lexer.next();
    } 
    else
      error("Error in literal value contruction");
    return value;
  }

  private boolean isBooleanOp() {
    return token.type().equals(TokenType.And) || token.type().equals(TokenType.Or);
  }

  private boolean isAddOp() {
    return token.type().equals(TokenType.Plus) || token.type().equals(TokenType.Minus);
  }

  private boolean isMultiplyOp() {
    return token.type().equals(TokenType.Multiply) || token.type().equals(TokenType.Divide);
  }

  private boolean isUnaryOp() {
    return token.type().equals(TokenType.Not) || token.type().equals(TokenType.Minus);
  }

  private boolean isEqualityOp() {
    return token.type().equals(TokenType.Equals) || token.type().equals(TokenType.NotEqual);
  }

  private boolean isRelationalOp() {
    return token.type().equals(TokenType.Less) || token.type().equals(TokenType.LessEqual)
        || token.type().equals(TokenType.Greater) || token.type().equals(TokenType.GreaterEqual);
  }

  private boolean isType() {
    return token.type().equals(TokenType.Int) || token.type().equals(TokenType.Bool)
        || token.type().equals(TokenType.Float) || token.type().equals(TokenType.Char);
  }

  private boolean isLiteral() {
    return token.type().equals(TokenType.IntLiteral) || isBooleanLiteral()
        || token.type().equals(TokenType.FloatLiteral)
        || token.type().equals(TokenType.CharLiteral);
  }

  private boolean isBooleanLiteral() {
    return token.type().equals(TokenType.True) || token.type().equals(TokenType.False);
  }

  private boolean isComma() {
    return token.type().equals(TokenType.Comma);
  }

  private boolean isSemicolon() {
    return token.type().equals(TokenType.Semicolon);
  }

  private boolean isLeftBrace() {
    return token.type().equals(TokenType.LeftBrace);
  }

  private boolean isRightBrace() {
    return token.type().equals(TokenType.RightBrace);
  }

  private boolean isStatement() {
    return isSemicolon() || isLeftBrace() || token.type().equals(TokenType.If)
        || token.type().equals(TokenType.While) || token.type().equals(TokenType.Identifier);
  }

  public static void main(String args[]) {
    Parser parser = new Parser(new Lexer("C:\\Users\\scenr\\eclipse-workspace\\Parser\\src\\parser\\p2.cl"));
    Program prog = parser.program();
    prog.display(0); // display abstract syntax tree
  } // main

} // Parser
