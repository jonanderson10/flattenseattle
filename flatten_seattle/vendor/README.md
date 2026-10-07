# Vendored third-party assets

`leaflet-1.9.4.min.js` and `leaflet-1.9.4.css` are Leaflet 1.9.4, obtained
from the npm registry (`https://registry.npmjs.org/leaflet/-/leaflet-1.9.4.tgz`,
retrieved 2026-09-16) and redistributed unmodified under the BSD 2-Clause
licence (Copyright Vladimir Agafonkin, CloudMade).

They are inlined into `outputs/sf_flat_routes_map.html` so that the
interactive map is a genuinely self-contained file: it opens and works with
no CDN and no network access, apart from the optional raster basemap tiles.
