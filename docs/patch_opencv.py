"""Patch opencv.js to force browser mode"""
import sys, os

path = sys.argv[1] if len(sys.argv) > 1 else "opencv.js"
with open(path, "rb") as f:
    data = f.read()

text = data.decode("utf-8", errors="ignore")

# Find the CommonJS branch and make the condition always false
old = '} else if (typeof module === \'object\' && module.exports) {'
new = '} else if (false && typeof module === \'object\' && module.exports) {'

if old in text:
    text = text.replace(old, new)
    print(f"OK: Patched UMD CommonJS branch")
else:
    # Try alternative: just replace module.exports = factory()
    old2 = 'module.exports = factory();'
    new2 = 'root.cv = factory();'
    if old2 in text:
        text = text.replace(old2, new2)
        print(f"OK: Replaced module.exports with root.cv")
    else:
        print("ERROR: Cannot find pattern to patch")
        sys.exit(1)

with open(path, "wb") as f:
    f.write(text.encode("utf-8"))

print(f"File size: {os.path.getsize(path)/1024/1024:.1f} MB")
