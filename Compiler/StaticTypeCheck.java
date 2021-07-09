// StaticTypeCheck.java

import java.util.*;

// Static type checking for Clite is defined by the functions 
// V and the auxiliary functions typing and typeOf.  These
// functions use the classes in the Abstract Syntax of Clite.


public class StaticTypeCheck {

    public static TypeMap typing (Declarations d) {
        TypeMap map = new TypeMap();
        for (Declaration di : d) 
            map.put (di.v, di.t);
        return map;
    }

    public static void check(boolean test, String msg) {
        if (test)  return;
        System.err.println(msg);
        //System.exit(1);
    }

    public static void V (Declarations d) {
        for (int i=0; i<d.size() - 1; i++)
            for (int j=i+1; j<d.size(); j++) {
                Declaration di = d.get(i);
                Declaration dj = d.get(j);
                check( ! (di.v.equals(dj.v)),
                       "duplicate declaration: " + dj.v);
            }
    } 

    public static void V (Program p) {
        V (p.decpart);

        V (p.body, typing (p.decpart));
    } 

    public static Type typeOf (Expression e, TypeMap tm) {
        if (e instanceof Value) return ((Value)e).type;
        if (e instanceof Variable) {
            Variable v = (Variable)e;
            check (tm.containsKey(v), "undefined variable: " + v);
            return (Type) tm.get(v);
        }
        if (e instanceof Binary) {
            Binary b = (Binary)e;
            if (b.op.ArithmeticOp( ))
                if (typeOf(b.term1,tm).equals(Type.FLOAT))
                    return (Type.FLOAT);
                else return (Type.INT);
            if (b.op.RelationalOp( ) || b.op.BooleanOp( )) 
                return (Type.BOOL);
        }
        if (e instanceof Unary) {
            Unary u = (Unary)e;
            if (u.op.NotOp( ))        return (Type.BOOL);
            else if (u.op.NegateOp( )) return typeOf(u.term,tm);
            else if (u.op.intOp( ))    return (Type.INT);
            else if (u.op.floatOp( )) return (Type.FLOAT);
            else if (u.op.charOp( ))  return (Type.CHAR);
        }
        throw new IllegalArgumentException("should never reach here");
    } 

    public static void V (Expression e, TypeMap tm) {
        if (e instanceof Value) 
            return;
        if (e instanceof Variable) { 
            Variable v = (Variable)e;
            check( tm.containsKey(v)
                   , "undeclared variable: " + v);
            return;
        }
        if (e instanceof Binary) {
            Binary b = (Binary) e;
            Type typ1 = typeOf(b.term1, tm);
            Type typ2 = typeOf(b.term2, tm);
            V (b.term1, tm);
            V (b.term2, tm);
            if (b.op.ArithmeticOp( ))  
                check((typ1.equals(typ2)) &&
                       (typ1.equals(Type.INT) || typ1.equals(Type.FLOAT))
                       , "type error for " + b.op);
            else if (b.op.RelationalOp( )) 
                check( typ1.equals(typ2) , "type error for " + b.op);
            else if (b.op.BooleanOp( )) 
                check( typ1.equals(Type.BOOL) && typ2.equals(Type.BOOL),
                       b.op + ": non-bool operand");
            else
                throw new IllegalArgumentException("should never reach here");
            return;
        }
		if (e instanceof Unary){
			Unary u = (Unary)e;
			Type typ1 = typeOf(u.term, tm);
			V(u.term, tm);
			if (u.op.NotOp()) {
				check(typ1.equals(Type.BOOL)
						, "type error for " + u.op);
			} else if (u.op.NegateOp()) {
				check((typ1.equals(Type.INT)) || (typ1.equals(Type.FLOAT))
						, "type error for " + u.op);
			} else if (u.op.floatOp() || u.op.charOp()){
				check(typ1.equals(Type.INT)
						, "type error for " + u.op);
			} else if (u.op.intOp()){
				check((typ1.equals(Type.FLOAT)) || (typ1.equals(Type.CHAR))
						, "type error for " + u.op);
			} else { 
				throw new IllegalArgumentException("should never reach here");
			}
			return;
		}
        throw new IllegalArgumentException("should never reach here");
    }

    public static void V (Statement s, TypeMap tm) {
        if ( s == null )
            throw new IllegalArgumentException( "AST error: null statement");
        if (s instanceof Skip) return;
        if (s instanceof Assignment) {
            Assignment a = (Assignment)s;
            check( tm.containsKey(a.target)
                   , " undefined target in assignment: " + a.target);
            V(a.source, tm);
            Type ttype = (Type)tm.get(a.target);
            Type srctype;
			if (tm.containsKey(a.source)){
				srctype = (Type)tm.get(a.source);
			} else {
				srctype = typeOf(a.source, tm);
			}
			if (!(ttype.equals(srctype))) {	//THIS HAD A BUG IN THEIR CODE AND WAS COMPARING OBJECTS INSTEAD OF VALUES
                if (ttype.equals(Type.FLOAT))
                    check( srctype.equals(Type.INT)
                           , "mixed mode assignment to " + a.target);
                else if (ttype.equals(Type.INT))
                    check( (srctype.equals(Type.CHAR)) || (srctype.equals(Type.FLOAT))
                           , "mixed mode assignment to " + a.target);
                else if (ttype.equals(Type.CHAR))
					check(srctype.equals(Type.INT)
						   	, "mixed mode assignment to " + a.target);
				else
                    check( false
                           , "mixed mode assignment to " + a.target);
            }
            return;
        } 
		if (s instanceof Conditional){
			Conditional c = (Conditional)s;
			check(typeOf(c.test, tm).equals(Type.BOOL)
					, "type error for " + c.test);
			V(c.test, tm);
			V(c.thenbranch, tm);
			if (c.elsebranch == null) return;
			V(c.elsebranch, tm);
			return;
		}
		if (s instanceof Loop){
			Loop l = (Loop)s;
			check(typeOf(l.test, tm).equals(Type.BOOL)
					, "type error for " + l.test);
			V(l.test, tm);
			V(l.body, tm);
			return;
		}
        if (s instanceof Block){
			Block b = (Block)s;
			for (Statement st : b.members){
				V(st, tm);
			}
			return;
		}
        if (s instanceof Print){
			Print p = (Print)s;
			V(p.source, tm);
			System.out.println(typeOf(p.source, tm));
			Type a = typeOf(p.source, tm);
			if(!a.equals(Type.INT) && !a.equals(Type.FLOAT)) {
				System.err.println("type error for" + p.source);
			}
			return;
		}
        if (s instanceof PrintCh){
        	PrintCh pCh = (PrintCh)s;
			V(pCh.source, tm);
			System.out.println(typeOf(pCh.source, tm));
			Type a = typeOf(pCh.source, tm);
			if(!a.equals(Type.CHAR)) {
				System.err.println("type error for" + pCh.source);
			}
			return;
		}
        throw new IllegalArgumentException("should never reach here");
    }

    public static void main(String args[]) {
        Parser parser  = new Parser(new Lexer(args[0]));
        Program prog = parser.program();
        prog.display();           // This also happens in Parser
        System.out.println("\nBegin type checking...");
        TypeMap map = typing(prog.decpart);
        map.display();   
        V(prog);
    } //main

} // class StaticTypeCheck

