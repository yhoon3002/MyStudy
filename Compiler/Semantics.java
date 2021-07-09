// Following is the semantics class:
// The meaning M of a Statement is a State
// The meaning M of a Expression is a Value

public class Semantics {

    State M (Program p) { 
        return M (p.body, initialState(p.decpart)); 
    }
  
    State initialState (Declarations d) {
        State state = new State();
        Value intUndef = new IntValue();
        for (Declaration decl : d)            state.put(decl.v, Value.mkValue(decl.t));
        return state;
    }
  
  
    State M (Statement s, State state) {
        if (s instanceof Skip) return M((Skip)s, state);
        if (s instanceof Assignment)  return M((Assignment)s, state);
        if (s instanceof Conditional)  return M((Conditional)s, state);
        if (s instanceof Loop)  return M((Loop)s, state);
        if (s instanceof Block)  return M((Block)s, state);
        if (s instanceof Print)  return M((Print)s, state);
        if (s instanceof PrintCh)  return M((PrintCh)s, state);
        throw new IllegalArgumentException("should never reach here");
    }
  
    State M (Skip s, State state) {
        return state;
    }
  
    State M (Assignment a, State state) {
        return state.onion(a.target, M (a.source, state));
    }
  
    State M (Block b, State state) {
        for (Statement s : b.members)
            state = M (s, state);
        return state;
    }
  
    State M (Conditional c, State state) {
        if (M(c.test, state).boolValue( ))
            return M (c.thenbranch, state);
        else
            return M (c.elsebranch, state);
    }
  
    State M (Loop l, State state) {
        if (M (l.test, state).boolValue( ))
            return M(l, M (l.body, state));
        else return state;
    }
    
    State M(Print p, State state) {
		System.out.println(M(p.source, state));
		return state;
	}

	State M(PrintCh pCh, State state) {
		if (M(pCh.source, state).toString().equals("\n")) System.out.print(M(pCh.source, state));
		else System.out.println(M(pCh.source, state));
		return state;
	}

    Value applyBinary (Operator op, Value v1, Value v2) {
        StaticTypeCheck.check( ! v1.isUndef( ) && ! v2.isUndef( ),
               "reference to undef value");
		//********** INT OPERATIONS ***********************
        if (op.val.equals(Operator.INT_PLUS)) 		
            return new IntValue(v1.intValue() + v2.intValue());
        if (op.val.equals(Operator.INT_MINUS)) 
            return new IntValue(v1.intValue() - v2.intValue());
        if (op.val.equals(Operator.INT_TIMES)) 
            return new IntValue(v1.intValue() * v2.intValue());
        if (op.val.equals(Operator.INT_DIV)) 
            return new IntValue(v1.intValue() / v2.intValue());
        if (op.val.equals(Operator.INT_LT))
			return new BoolValue(v1.intValue() < v2.intValue());
		if (op.val.equals(Operator.INT_LE))
			return new BoolValue(v1.intValue() <= v2.intValue());
		if (op.val.equals(Operator.INT_EQ))
			return new BoolValue(v1.intValue() == v2.intValue());
		if (op.val.equals(Operator.INT_NE))
			return new BoolValue(v1.intValue() != v2.intValue());
		if (op.val.equals(Operator.INT_GT))
			return new BoolValue(v1.intValue() > v2.intValue());
		if (op.val.equals(Operator.INT_GE))
			return new BoolValue(v1.intValue() >= v2.intValue());
		//*********** FLOAT OPERATIONS ***********************
		if (op.val.equals(Operator.FLOAT_LT))
			return new BoolValue(v1.floatValue() < v2.floatValue());
		if (op.val.equals(Operator.FLOAT_LE))
			return new BoolValue(v1.floatValue() <= v2.floatValue());
		if (op.val.equals(Operator.FLOAT_EQ))
			return new BoolValue(v1.floatValue() == v2.floatValue());
		if (op.val.equals(Operator.FLOAT_NE))
			return new BoolValue(v1.floatValue() != v2.floatValue());
		if (op.val.equals(Operator.FLOAT_GT))
			return new BoolValue(v1.floatValue() > v2.floatValue());
		if (op.val.equals(Operator.FLOAT_GE))
			return new BoolValue(v1.floatValue() >= v2.floatValue());
		if (op.val.equals(Operator.FLOAT_PLUS))
			return new FloatValue(v1.floatValue() + v2.floatValue());
		if (op.val.equals(Operator.FLOAT_MINUS))
			return new FloatValue(v1.floatValue() - v2.floatValue());
		if (op.val.equals(Operator.FLOAT_TIMES))
			return new FloatValue(v1.floatValue() * v2.floatValue());
		if (op.val.equals(Operator.FLOAT_DIV))
			return new FloatValue(v1.floatValue() / v2.floatValue());
		//************* CHAR OPERATIONS *************************
		if (op.val.equals(Operator.CHAR_LT))
			return new BoolValue(v1.charValue() < v2.charValue());
		if (op.val.equals(Operator.CHAR_LE))
			return new BoolValue(v1.charValue() <= v2.charValue());
		if (op.val.equals(Operator.CHAR_EQ))
			return new BoolValue(v1.charValue() == v2.charValue());
		if (op.val.equals(Operator.CHAR_NE))
			return new BoolValue(v1.charValue() != v2.charValue());
		if (op.val.equals(Operator.CHAR_GT))
			return new BoolValue(v1.charValue() > v2.charValue());
		if (op.val.equals(Operator.CHAR_GE))
			return new BoolValue(v1.charValue() >= v2.charValue());
		//************* BOOL OPERATIONS ************************
		if (op.val.equals(Operator.BOOL_LT))
			throw new IllegalArgumentException("Operator '<' not supported for type bool");
		if (op.val.equals(Operator.BOOL_LE))
			throw new IllegalArgumentException("Operator '<=' not supported for type bool");
		if (op.val.equals(Operator.BOOL_EQ))
			return new BoolValue(v1.boolValue() == v2.boolValue());
		if (op.val.equals(Operator.BOOL_NE))
			return new BoolValue(v1.boolValue() != v2.boolValue());
		if (op.val.equals(Operator.BOOL_GT))
			throw new IllegalArgumentException("Operator '>' not supported for type bool");
		if (op.val.equals(Operator.BOOL_GE))
			throw new IllegalArgumentException("Operator '>=' not supported for type bool");
		if (op.val.equals(Operator.BOOL_AND))
			return new BoolValue(v1.boolValue() && v2.boolValue());
		if (op.val.equals(Operator.BOOL_OR))
			return new BoolValue(v1.boolValue() || v2.boolValue());
        throw new IllegalArgumentException("should never reach here");
    } 
    
    Value applyUnary (Operator op, Value v) {
        StaticTypeCheck.check( ! v.isUndef( ),
               "reference to undef value");
        if (op.val.equals(Operator.BOOL_NOT))
            return new BoolValue(!v.boolValue( ));
        else if (op.val.equals(Operator.INT_NEG))
            return new IntValue(-v.intValue( ));
        else if (op.val.equals(Operator.FLOAT_NEG))
            return new FloatValue(-v.floatValue( ));
        else if (op.val.equals(Operator.I2F))
            return new FloatValue((float)(v.intValue( ))); 
        else if (op.val.equals(Operator.F2I))
            return new IntValue((int)(v.floatValue( )));
        else if (op.val.equals(Operator.C2I))
            return new IntValue((int)(v.charValue( )));
        else if (op.val.equals(Operator.I2C))
            return new CharValue((char)(v.intValue( )));
        throw new IllegalArgumentException("should never reach her"
        		+ "e");
    } 

    Value M (Expression e, State state) {
        if (e instanceof Value) 
            return (Value)e;
        if (e instanceof Variable) 
            return (Value)(state.get(e));
        if (e instanceof Binary) {
            Binary b = (Binary)e;
            return applyBinary (b.op, 
                                M(b.term1, state), M(b.term2, state));
        }
        if (e instanceof Unary) {
            Unary u = (Unary)e;
            return applyUnary(u.op, M(u.term, state));
        }
        throw new IllegalArgumentException("should never reach here");
    }

    public static void main(String args[]) {
        Parser parser  = new Parser(new Lexer(args[0]));
        Program prog = parser.program();
		System.out.println("Begin parsing..." + args[0]);
        prog.display();    // This was done in the Parser
        System.out.println("\nBegin type checking..." + args[0]);
        TypeMap map = StaticTypeCheck.typing(prog.decpart);
        map.display();    // This was done in the TypeMap
        StaticTypeCheck.V(prog);
        Program out = TypeTransformer.T(prog, map);
        System.out.println("\nTransformed AST:\n");
        out.display();    // Transformed ast
        Semantics semantics = new Semantics( );
        State state = semantics.M(out);
		System.out.println("\nBegin interpreting..." + args[0]);
        System.out.println("Final State");
        state.display( );  // Done in State.java
    }
}
