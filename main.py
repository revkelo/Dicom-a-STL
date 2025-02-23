import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import pydicom
import numpy as np
from skimage import measure, morphology
from scipy import ndimage
from stl import mesh
import trimesh
from trimesh.smoothing import filter_laplacian

# Crear ventana principal
root = tk.Tk()
root.title("DICOM a STL Reducido")
root.geometry("400x250")

status_label = tk.Label(root, text="Selecciona una carpeta con archivos DICOM o un archivo STL", wraplength=300)
status_label.pack(pady=20)

def procesar_dicom(folder_path):
    try:
        status_label.config(text="Cargando archivos DICOM...")
        dicom_files = [pydicom.dcmread(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith('.dcm')]
        dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]) if 'ImagePositionPatient' in x else 0)
        
        slices = np.stack([s.pixel_array for s in dicom_files])
        rescale_slope = float(getattr(dicom_files[0], 'RescaleSlope', 1))
        rescale_intercept = float(getattr(dicom_files[0], 'RescaleIntercept', 0))
        slices = slices * rescale_slope + rescale_intercept
        
        pixel_spacing = dicom_files[0].PixelSpacing
        slice_thickness = float(getattr(dicom_files[0], 'SliceThickness', 1))
        spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
        
        threshold_min, threshold_max = 150, 3075
        binary_volume = np.logical_and(slices > threshold_min, slices < threshold_max)
        binary_volume = morphology.remove_small_objects(binary_volume, min_size=1000)
        smoothed_volume = ndimage.gaussian_filter(binary_volume.astype(float), sigma=1)
        
        verts, faces, normals, _ = measure.marching_cubes(smoothed_volume, level=0.5, spacing=spacing)
        centroid = np.mean(verts, axis=0)
        centered_verts = verts - centroid
        
        rotation_matrix = np.array([
            [np.cos(np.radians(90)), 0, np.sin(np.radians(90))],
            [0, 1, 0],
            [-np.sin(np.radians(90)), 0, np.cos(np.radians(90))]
        ])
        rotated_verts = np.dot(centered_verts, rotation_matrix)
        
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = rotated_verts[f[j], :]
        
        output_path = os.path.join(folder_path, "modelo.stl")
        stl_mesh.save(output_path)
        
        status_label.config(text="Reduciendo malla...")
        original_mesh = trimesh.load_mesh(output_path)
        reduction_faces = int(original_mesh.faces.shape[0] * 0.1)
        reduced_mesh = original_mesh.simplify_quadratic_decimation(reduction_faces)
        smoothed_reduced_mesh = reduced_mesh.copy()
        filter_laplacian(smoothed_reduced_mesh, lamb=0.5, iterations=10)
        
        reduced_output_path = os.path.join(folder_path, "modelo-reducido.stl")
        smoothed_reduced_mesh.export(reduced_output_path)
        
        status_label.config(text="STL reducido generado exitosamente.")
        messagebox.showinfo("Éxito", f"Modelo STL reducido guardado en:\n{reduced_output_path}")
        
        smoothed_reduced_mesh.show(background=[0, 0, 0, 1], smooth=False, color=[1, 1, 1, 1])


    except Exception as e:
        status_label.config(text="Error durante el procesamiento.")
        messagebox.showerror("Error", str(e))

def mostrar_stl():
    file_path = filedialog.askopenfilename(filetypes=[("STL files", "*.stl")])
    if file_path:
        try:
            status_label.config(text="Cargando archivo STL...")
            stl_mesh = trimesh.load_mesh(file_path)
            stl_mesh.show(background=[0, 0, 0, 1], smooth=False, color=[1, 1, 1, 1])
            status_label.config(text="Archivo STL mostrado correctamente.")
        except Exception as e:
            status_label.config(text="Error al mostrar el STL.")
            messagebox.showerror("Error", str(e))

def seleccionar_carpeta():
    folder_path = filedialog.askdirectory()
    if folder_path:
        status_label.config(text="Procesando carpeta seleccionada...")
        threading.Thread(target=procesar_dicom, args=(folder_path,)).start()

# Botones para seleccionar carpeta DICOM o archivo STL
btn_select_folder = tk.Button(root, text="Seleccionar Carpeta DICOM", command=seleccionar_carpeta)
btn_select_folder.pack(pady=10)

btn_select_stl = tk.Button(root, text="Seleccionar y Mostrar STL", command=mostrar_stl)
btn_select_stl.pack(pady=10)

# Ejecutar la ventana
root.mainloop()
