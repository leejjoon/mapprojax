from .converters import WCSConverter
from asdf.extension import ManifestExtension
from asdf.resource import DirectoryResourceMapping
import os

def get_resource_mappings():
    resources_root = os.path.join(os.path.dirname(__file__), '../../resources')
    
    return [
        DirectoryResourceMapping(
            resources_root, 
            "http://mapprojax.org/schemas", # This prefix maps to the schemas dir?
            # Actually, standard is to map to the root URL
            # id: http://mapprojax.org/schemas/wcs-1.0.0
            # file: .../schemas/wcs-1.0.0.yaml
        ),
        # Manifests usually need their own mapping or live under the same root
        DirectoryResourceMapping(
            resources_root,
            "http://mapprojax.org/manifests",
        )
    ]

def get_extensions():
    converters = [WCSConverter()]
    return [
        ManifestExtension.from_uri(
            "http://mapprojax.org/manifests/mapprojax-1.0.0",
            converters=converters,
        )
    ]