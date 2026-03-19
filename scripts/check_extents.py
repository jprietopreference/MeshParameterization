import struct, json, os
os.chdir("d:/RnD/MeshParameterization")
meshes = ["cube_100mm","sphere_R100mm","torus_R50mm_r15mm","torus_R200mm_r40mm",
          "teapot","klein_bottle_occ","cgalref_nefertiti","cgalref_three_peaks",
          "cgalref_head","cgalref_mushroom"]
for f in meshes:
    p = f"data/{f}.glb"
    if not os.path.exists(p):
        continue
    with open(p, "rb") as fp:
        d = fp.read()
    jl = struct.unpack("<I", d[12:16])[0]
    g = json.loads(d[20:20+jl])
    a = g["accessors"][0]
    mn, mx = a["min"], a["max"]
    ext = [mx[i] - mn[i] for i in range(3)]
    n = a["count"]
    print(f"{f:30s}  {ext[0]:8.2f} x {ext[1]:8.2f} x {ext[2]:8.2f}   ({n} verts)")
