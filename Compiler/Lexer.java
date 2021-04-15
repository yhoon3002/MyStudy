package lexer;
import java.io.*;

public class Lexer {

    private boolean isEof = false;
    private char ch = ' '; // 현재 character 
    private BufferedReader input; // input을 읽는데 사용됨
    private String line = ""; // 현재 line
    private int lineno = 0; // 현재 file의 line
    private int col = 1; // 현재 file의 column 수
    private final String letters = "abcdefghijklmnopqrstuvwxyz"
        + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // 모든 가능한 글자
    private final String digits = "0123456789"; // 모든 가능한 수
    private final char eolnCh = '\n'; // line의 끝
    private final char eofCh = '\004'; // file의 끝
    

    public Lexer (String fileName) { // source filename
        try {
            input = new BufferedReader (new FileReader(fileName));
        }
        catch (FileNotFoundException e) {
            System.out.println("File not found: " + fileName);
            System.exit(1);
        }
    }

    private char nextChar() { // Return next char
        if (ch == eofCh)
            error("Attempt to read past end of file");
        col++;
        if (col >= line.length()) {
            try {
                line = input.readLine( );
            } catch (IOException e) {
                System.err.println(e);
                System.exit(1);
            } // try
            if (line == null) // at end of file
                line = "" + eofCh;
            else {
                 System.out.println("Line " + lineno + ":\t" + line);
                lineno++;
                line += eolnCh;
            } // if line
            col = 0;
        } // if col
        return line.charAt(col);
    }
            
    // 만약 현재 character가 letter 이라면 계속 가다가 
	// letter 또는 digit이 아닌 것을 만나면, keyword로 return 해준다.
    // 마찬가지로, 만약 현재 character가 digit 이라면 계속 가다가
    // digit이 아닌 것을 만난다면 . 인지를 확인하고 
    // .이라면 소수점을 만들게 된다.
    // 또한 ' ', '\t', '\r', '\EOL'은 건너 뛴다.
    public Token next( ) { // Return next token
        do {
            if (isLetter(ch)) { // ident or keyword
                String spelling = concat(letters + digits);
                return Token.keyword(spelling);
            }
            else if (isDigit(ch)) { // int or float literal
                String number = concat(digits);
                if (ch != '.')  // int Literal
                    return Token.mkIntLiteral(number);
                number += concat(digits);
                return Token.mkFloatLiteral(number);
            }
            else switch (ch) {
            case ' ': case '\t': case '\r': case eolnCh:
                ch = nextChar();
                break;
            
            case '/':  // divide or comment
                ch = nextChar();
                if (ch != '/')  return Token.divideTok;
                // comment
                do {
                    ch = nextChar();
                } while (ch != eolnCh);
                ch = nextChar();
                break;
            
            case '\'':  // char literal
                char ch1 = nextChar();
                nextChar(); // get '
                ch = nextChar();
                return Token.mkCharLiteral("" + ch1);
                
            case eofCh: return Token.eofTok;
            
            case '+': ch = nextChar();
                return Token.plusTok;
                
            // - * ( ) { } ; ,  student exercise
            case '-': ch = nextChar();
                return Token.minusTok;

            case '*': ch = nextChar();
                return Token.multiplyTok;

            case '(': ch = nextChar();
                return Token.leftParenTok;
            
            case ')': ch = nextChar();
                return Token.rightParenTok;

            case '{': ch = nextChar();
                return Token.leftBraceTok;
            
            case '}': ch = nextChar();
                return Token.rightBraceTok;

            case ';': ch = nextChar();
                return Token.semicolonTok;
            
            case ',': ch = nextChar();
                return Token.commaTok;
                
            case '&': check('&'); return Token.andTok;
            
            case '|': check('|'); return Token.orTok;

            case '=':
                return chkOpt('=', Token.assignTok,
                                   Token.eqeqTok);

            // < > !  student exercise 
            case '<':
                return chkOpt('<', Token.ltTok, Token.lteqTok);
            
            case '>':
                return chkOpt('>', Token.gtTok, Token.gteqTok);
            
            case '!':
                return chkOpt('!', Token.notTok, Token.noteqTok);

            default:  error("Illegal character " + ch); 
            } // switch
        } while (true);
    } // next

    // c가 letter인지 아닌지 확인하는 것이다.
    private boolean isLetter(char c) {
        return (c>='a' && c<='z' || c>='A' && c<='Z');
    }
    
    // c가 digit인지 아닌지 확인하는 것이다.
    private boolean isDigit(char c) {
        // student exercise
        return (c >= '0' && c <= '9');
    }
    
    // ch가 주어진 character인지 확인하는 것이다.
    // 만약 주어진 character가 아니라면, 에러를 띄운다.
    private void check(char c) {
        ch = nextChar();
        if (ch != c) 
            error("Illegal character, expecting " + c);
        ch = nextChar();
    }
    
    // c가 two의 값인지 확인하고
    // 맞다면, 현재 character를 건너뛰고, two를 return한다.
    // 아니라면, one을 return한다.
    private Token chkOpt(char c, Token one, Token two) {
        // student exercise
        ch = nextChar();
        if (ch != 'c')
        {
            ch = nextChar();
            return one;
        }

        else 
        {
            ch = nextChar();
            return two;
        }
    }
    
    // 주어진 set에 없는 문자가 있을 때까지 character를 추가한다.
    private String concat(String set) {
        String r = "";
        do {
            r += ch;
            ch = nextChar();
        } while (set.indexOf(ch) >= 0);
        return r;
    }
    
    // error를 출력한다.
    public void error (String msg) {
        System.err.print(line);
        System.err.println("Error: column " + col + " " + msg);
        System.exit(1);
    }
    
    static public void main ( String[] argv ) {
        Lexer lexer = new Lexer("C:/Users/scenr/eclipse-workspace/Lexer/src/lexer/test2.txt");
        Token tok = lexer.next( );
        while (tok != Token.eofTok) {
            System.out.println(tok.toString());
            tok = lexer.next( );
        } 
    } // main
}
