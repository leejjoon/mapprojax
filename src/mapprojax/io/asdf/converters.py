from asdf.extension import Converter
import mapprojax

class WCSConverter(Converter):
    tags = ["http://mapprojax.org/tags/wcs-1.0.0"]
    types = [
        mapprojax.Tan, 
        mapprojax.Sin, 
        mapprojax.TanArray, 
        mapprojax.SinArray
    ]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            'crpix': obj.crpix,
            'cd': obj.cd,
            'projection_class': obj.__class__.__name__
        }
        if hasattr(obj, 'crvals'):
             # It's an array class
             # obj.crvals is a tuple of (ra_arr, dec_arr)
             # We can store it as a list
             node['crval'] = list(obj.crvals)
        else:
             node['crval'] = obj.crval
             
        return node

    def from_yaml_tree(self, node, tag, ctx):
        cls_name = node['projection_class']
        cls = getattr(mapprojax, cls_name)
        
        return cls(node['crpix'], node['cd'], node['crval'])
