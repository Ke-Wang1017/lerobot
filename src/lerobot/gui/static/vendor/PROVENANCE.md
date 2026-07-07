# Vendored browser libraries

These files are unmodified third-party JavaScript, vendored so the URDF
visualization (`static/urdf_viz.html`) runs offline with no CDN dependency.

## three.js — r169

- Source: https://github.com/mrdoob/three.js (`build/three.module.js` and
  `examples/jsm/...`)
- License: MIT
- Files: `three.module.js`, `OrbitControls.js`, `STLLoader.js`,
  `ColladaLoader.js`, `TGALoader.js`

## urdf-loader

- Source: https://github.com/gkjohnson/urdf-loaders (`src/urdf-loader/`)
- License: Apache-2.0
- Files: `URDFLoader.js`, `URDFClasses.js`

## Local modifications

- `ColladaLoader.js`: the `TGALoader` import path was changed from
  `'../loaders/TGALoader.js'` to `'./TGALoader.js'` so all vendored modules
  resolve from this flat directory.
