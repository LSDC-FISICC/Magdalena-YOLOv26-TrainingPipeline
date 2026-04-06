from pathlib import Path
from PIL import Image
import imagehash

folder = Path("imagenes_labelbox")

# Paso 1: calcular hashes de todas las imágenes
print("Calculando hashes...")
hashes = {}
duplicates = []

for img_path in sorted(folder.glob("*.png")):
    try:
        ph = imagehash.phash(Image.open(img_path))
        
        matched = False
        for existing_hash, existing_path in hashes.items():
            if abs(ph - existing_hash) <= 4:  # threshold 4 = casi idénticas
                duplicates.append(img_path)
                matched = True
                break
        
        if not matched:
            hashes[ph] = img_path

    except Exception as e:
        print(f"Error con {img_path.name}: {e}")

# Paso 2: reporte antes de borrar
print(f"\nImágenes únicas:     {len(hashes)}")
print(f"Imágenes duplicadas: {len(duplicates)}")
print(f"Total:               {len(hashes) + len(duplicates)}")

# Paso 3: borrar (solo correr si el reporte se ve bien)
for dup in duplicates:
    dup.unlink()
    print(f"Borrado: {dup.name}")

print(f"\nEliminadas {len(duplicates)} imágenes duplicadas.")