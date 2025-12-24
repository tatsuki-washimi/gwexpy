import inspect
import os
import sys
import importlib

# Ensure current directory is in path
sys.path.insert(0, os.getcwd())

import gwexpy.timeseries
import gwexpy.frequencyseries
import gwexpy.spectrogram
import gwexpy.types
import gwexpy.plot

# List of classes to document
CLASSES_TO_DOCUMENT = [
    gwexpy.timeseries.TimeSeries,
    gwexpy.timeseries.TimeSeriesDict,
    gwexpy.timeseries.TimeSeriesList,
    gwexpy.timeseries.TimeSeriesMatrix,
    gwexpy.timeseries.Transform,
    gwexpy.timeseries.Pipeline,
    gwexpy.timeseries.ImputeTransform,
    gwexpy.timeseries.StandardizeTransform,
    gwexpy.timeseries.WhitenTransform,
    gwexpy.timeseries.PCATransform,
    gwexpy.timeseries.ICATransform,
    gwexpy.frequencyseries.FrequencySeries,
    gwexpy.frequencyseries.FrequencySeriesDict,
    gwexpy.frequencyseries.FrequencySeriesList,
    gwexpy.frequencyseries.FrequencySeriesMatrix,
    gwexpy.spectrogram.Spectrogram,
    gwexpy.spectrogram.SpectrogramDict,
    gwexpy.spectrogram.SpectrogramList,
    gwexpy.spectrogram.SpectrogramMatrix,
    gwexpy.types.Array2D,
    gwexpy.types.Plane2D,
    gwexpy.types.SeriesMatrix,
    gwexpy.types.TimePlaneTransform,
    gwexpy.plot.Plot,
]

OUTPUT_DIR = '/home/washimi/work/gwexpy/docs/api-reference'

def get_method_signature(method):
    try:
        sig = inspect.signature(method)
        return str(sig)
    except ValueError:
        return "(...)"

def format_docstring(doc):
    if not doc:
        return ""
    # Remove leading indentation
    lines = doc.split('\n')
    if not lines:
        return ""
    
    # First line often has no indent
    formatted = [lines[0]]
    if len(lines) > 1:
        # Find min indent for rest
        indents = [len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()]
        if indents:
            min_indent = min(indents)
            for line in lines[1:]:
                formatted.append(line[min_indent:])
        else:
            formatted.extend(lines[1:])
            
    return '\n'.join(formatted)

def should_document_member(name, member, cls):
    if name.startswith('_') and name != '__init__':
        return False
        
    obj_to_check = member
    if isinstance(member, property):
        obj_to_check = member.fget
        
    if not callable(obj_to_check) and not isinstance(member, property):
        return False

    # Check module origin
    try:
        module = inspect.getmodule(obj_to_check)
        if module:
            mod_name = module.__name__
            
            # Include gwexpy and gwpy methods.
            if mod_name.startswith('gwexpy'):
                return True
            if mod_name.startswith('gwpy'):
                return True
            
            # Special case for mixins or dynamically created methods that might appear as belonging to the class module
            # If the methods belong to any of the mixin classes in gwexpy, they should have gwexpy module.
            
            return False
        else:
            # Module is None. This happens for:
            # 1. Built-in functions/methods (e.g. numpy.ndarray methods from C-extension)
            # 2. Some properties if not set up correctly (though fget usually has module)
            
            # We assume anything with module=None is NOT part of the python API we want to document (likely C-backend stuff)
            # unless we can verify it's defined in the class source.
            
            # But wait, sometimes things defined in __main__ or weird places have None.
            # Given we are importing from installed packages / local files, pure python methods should have modules.
            # So excluding None is safe for removing numpy C methods.
            return False
            
    except:
        pass
        
    return False

def get_docstring(cls, name, member):
    """
    Get docstring for a member, checking MRO for inherited docstrings
    if the member itself has none.
    """
    # 1. Check direct docstring
    doc = getattr(member, "__doc__", None)
    if isinstance(member, property):
        if member.fget and member.fget.__doc__:
            doc = member.fget.__doc__

    if doc:
        return format_docstring(doc)

    # 2. Check MRO for inherited docstring
    # cls.__mro__ includes cls itself, then bases.
    # We want to skip cls itself if we already checked it, but `getattr(member)` checked the bound member.
    # Iterate bases.
    for base in cls.__mro__[1:]:
        if hasattr(base, name):
            base_member = getattr(base, name)
            base_doc = getattr(base_member, "__doc__", None)
            
            # Handle properties in base
            if isinstance(base_member, property):
                if base_member.fget:
                    base_doc = base_member.fget.__doc__
                
            if base_doc:
                # Add a note that it is inherited
                return format_docstring(base_doc) + f"\n\n*(Inherited from `{base.__name__}`)*"

    return ""

def generate_markdown_for_class(cls):
    class_name = cls.__name__
    md_content = f"# {class_name}\n\n"
    
    # Inheritance
    bases = [b.__name__ for b in cls.__bases__]
    md_content += f"**Inherits from:** {', '.join(bases)}\n\n"
    
    # Class Docstring
    doc = format_docstring(cls.__doc__)
    if doc:
        md_content += f"{doc}\n\n"
    
    md_content += "## Methods\n\n"
    
    members = inspect.getmembers(cls)
    # Sort members: __init__ first, then alphabetical
    members.sort(key=lambda x: (0 if x[0] == '__init__' else 1, x[0]))
    
    for name, member in members:
        if should_document_member(name, member, cls):
            md_content += f"### `{name}`\n\n"
            
            if callable(member):
                sig = get_method_signature(member)
                md_content += f"```python\n{name}{sig}\n```\n\n"
            
            # Use get_docstring to handle inheritance
            member_doc = get_docstring(cls, name, member)
            if member_doc:
                md_content += f"{member_doc}\n\n"
            else:
                md_content += "_No documentation available._\n\n"
                
    return md_content

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for cls in CLASSES_TO_DOCUMENT:
        print(f"Generating docs for {cls.__name__}...")
        try:
            md = generate_markdown_for_class(cls)
            filename = os.path.join(OUTPUT_DIR, f"{cls.__name__}.md")
            with open(filename, 'w') as f:
                f.write(md)
        except Exception as e:
            print(f"Failed to generate docs for {cls.__name__}: {e}")

if __name__ == "__main__":
    main()
