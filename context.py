"""
This module implements functions to manage the context.
For example, to retrieve the value of variables 
from the user's namespace and provide detailed information about them.
"""

import inspect
import types
from typing import Any, Dict


def filter_variables(namespace: Dict[str, Any]):
    """Filter the list of non-private variables in the current Jupyter Notebook session."""
    variables_and_values = {}
    for name, value in namespace.items():
        # Exclude internal variables and modules
        if not name.startswith("_") and not isinstance(value, types.ModuleType):
            # Exclude IPython's In and Out history
            if name not in ["In", "Out"]:
                # Exclude functions and other callables
                if not callable(value):
                    variables_and_values[name] = value
    # return a dict ({k:v})
    return variables_and_values


def extract_variables_from_query(line: str) -> set:
    """Extract potential variable names from the query string."""
    variables = set()

    for word in line.split():
        # Only consider words that look like valid Python identifiers
        if word.isidentifier():
            variables.add(word)
    return variables


def get_variable_info(name: str, value: Any) -> str:
    """Get detailed information about a variable."""
    # TODO improve, remove functions, simplify
    info_parts = [f"Variable: {name}"]
    info_parts.append(f"Type: {type(value).__name__}")

    # dataframes
    if "pandas.core.frame.DataFrame" in str(type(value)):
        info_parts.append(f"Shape: {value.shape}")
        info_parts.append("Columns:")
        for col in value.columns:
            info_parts.append(f"- {col} ({value[col].dtype})")
        info_parts.append("\nSample (first 5 rows):")
        info_parts.append(str(value.head()))

    # functions
    elif inspect.isfunction(value):
        info = [
            f"Variable: {name}",
            "Type: function",
            (
                f"Documentation: {inspect.getdoc(value)}"
                if inspect.getdoc(value)
                else None
            ),
            "Source code:",
            inspect.getsource(value),
        ]
        return "\n".join(filter(None, info))

    # objects
    elif hasattr(value, "__dict__"):
        attrs = dir(value)
        info_parts.append("Attributes:")
        for attr in attrs:
            if not attr.startswith("_"):
                info_parts.append(f"- {attr}")

    # containers
    elif hasattr(value, "__len__"):
        info_parts.append(f"Length: {len(value)}")
        try:
            sample = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            info_parts.append(f"Sample: {sample}")
        except Exception:
            pass

    # any other object
    else:
        try:
            info_parts.append(f"Type: {type(value).__name__}")

            sample = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            info_parts.append(f"\nString representation: {sample}")

            if value.__doc__:
                doc = value.__doc__.strip()
                info_parts.append("Documentation:")
                info_parts.append(doc[:500] + "..." if len(doc) > 500 else doc)

        except Exception:
            pass

    return "\n".join(info_parts)


def get_context(namespace: Dict[str, Any], query: str) -> str:
    """Extract relevant context from the user's namespace based on the query."""
    # Filter namespace to only include actual variables (non-private)
    filtered_namespace = {k: v for k, v in namespace.items() if not k.startswith("_")}

    mentioned_vars = extract_variables_from_query(query)
    context_parts = []

    for var_name in mentioned_vars:
        if var_name in filtered_namespace:
            var = filtered_namespace[var_name]
            context_parts.append(get_variable_info(var_name, var))

    return "\n".join(context_parts)
