# Indoor Mapping

Prject file structure:
---------------------

```
|  
└───README.md  
|  
└───data/  
|    | color_images/  
|    | depth_images/  
|    | camera_intrinsics/  
|  
└───src/  
|    | main.py  
|    | imageData.py  
```

General file naming conventions for matterport images:
---------------------

The Matterport Pro camera consists of three Primesense Carmine depth and RGB
sensors placed on an eye-height tripod.   The
three cameras (camera index 0-2) are oriented diagonally up, flat, and
diagonally down, cumulatively covering most of the vertical field of view.
During capture, the camera rotates  around its vertical axis and stops at
60-degree intervals (yaw index 0-5). So, for each tripod placement
(panorama_uuid), a total of 18 images are captured.  Files associated with
these images adhere to the following naming convention:

    <panorama_uuid>_<imgtype><camera_index>_<yaw_index>.<extension>

where <panorama_uuid> is a unique string, <camera_index> is [0-5], and
<yaw_index> is [0-2].  <imgtype> is 'j' for HDR images, 'i' for tone-mapped
color images, 'd' for depth images, "skybox" for skybox images, "pose" for
camera pose files, and "intrinsics" for camera intrinsics files.  The
extension is ".jxr" for HDR images, ".jpg" for tone-mapped color images,
and ".png" for depth and normal images.


matterport camera intrinsics
---------------------

Intrinsic parameters for every camera, stored as ASCII in the following format
(using OpenCV's notation):

    width height fx fy cx cy k1 k2 p1 p2 k3

The Matterport camera axis convention is:
    x-axis: right
    y-axis: down
    z-axis: "look"
