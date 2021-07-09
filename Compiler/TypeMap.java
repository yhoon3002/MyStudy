import java.util.*;

public class TypeMap extends HashMap<Variable, Type> { 

// TypeMap is implemented as a Java HashMap.  
// Plus a 'display' method to facilitate experimentation.

	public void display(){
		System.out.println("Type map: {");
		for(Map.Entry<Variable, Type> entry : this.entrySet()){
			Variable v = entry.getKey();
			Type t = entry.getValue();
			System.out.println("  <" + v + ", " + t + ">");
		}
		System.out.println("}");
	}
}
