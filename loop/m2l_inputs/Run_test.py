from map2loop.project import Project 
from map2loop.m2l_enums import VerboseLevel 
import shutil
import os

proj = Project(
                geology_filename='../inputs/geology_coded.shp',
                fault_filename='../inputs/faults_dense.shp',
                fold_filename='../inputs/faults.shp',
                structure_filename='../inputs/bedding.shp',
                mindep_filename='http://13.211.217.129:8080/geoserver/loop/wfs?service=WFS&version=1.0.0&request=GetFeature&typeName=loop:null_mindeps&bbox={BBOX_STR}&srs=EPSG:28350&outputFormat=shape-zip',
                dtm_filename='../inputs/Moulouya_dtm_small.tif',
                metadata_filename='./data.json',
                overwrite='false',
                verbose_level=VerboseLevel.NONE,
                project_path='../m2l_outputs',
                working_projection='epsg:26191',
                )
 
proj.update_config(
                    out_dir='../m2l_outputs',
                    bbox_3d={'minx': 710718, 'miny': 229525, 'maxx': 739773, 'maxy': 251976, 'base': -2900, 'top': 2100},
                    run_flags={'aus': True, 'close_dip': -999.0, 'contact_decimate': 0, 'contact_dip': -999.0, 'contact_orientation_decimate': 5, 'deposits': 'Fe,Cu,Au,NONE', 'dist_buffer': 10.0, 'dtb': '', 'fat_step': 750.0, 'fault_decimate': 5, 'fault_dip': 90.0, 'fold_decimate': 5, 'interpolation_scheme': 'scipy_rbf', 'interpolation_spacing': 500.0, 'intrusion_mode': 0, 'max_thickness_allowed': 10000.0, 'min_fault_length': 2000.0, 'misorientation': 30.0, 'null_scheme': 'null_scheme', 'orientation_decimate': 0, 'pluton_dip': 45.0, 'pluton_form': 'domes', 'thickness_buffer': 5000.0, 'use_fat': False, 'use_interpolations': False, 'fault_orientation_clusters': 2, 'fault_length_clusters': 2, 'use_roi_clip': False, 'roi_clip_path': '','ignore_codes':['q']},
                    proj_crs='epsg:26191',
                    clut_path='',
                )

if(not os.path.isdir(proj.config.project_path+'/MINERAL_DEPOSIT')):
    os.mkdir(proj.config.project_path+'/MINERAL_DEPOSIT')
if(not os.path.isdir(proj.config.project_path+'/FAULT')):
    os.mkdir(proj.config.project_path+'/FAULT')
if(not os.path.isdir(proj.config.project_path+'/GEOLOGY')):
    os.mkdir(proj.config.project_path+'/GEOLOGY')
if(not os.path.isdir(proj.config.project_path+'/STRUCTURE')):
    os.mkdir(proj.config.project_path+'/STRUCTURE')
if(not os.path.isdir(proj.config.project_path+'/FOLD')):
    os.mkdir(proj.config.project_path+'/FOLD')
proj.workflow['fold_axial_traces']=False
proj.workflow['contact_dips'] = True
proj.run()
proj.map_data.save_all_map_data( proj.config.project_path, extension = '.csv')
proj.save_mapdata_to_shapefiles(proj.config.project_path+'/maroc_')
#shutil.copyfile('../source_data/map2loop.qgz', proj.config.project_path+'/map2loop.qgz')


proj_path = proj.config.project_path
graph_path = proj.config.graph_path
tmp_path = proj.config.tmp_path
dtm_path = proj.config.project_path+'/dtm'
output_path = proj.config.output_path
vtk_path = proj.config.project_path+'/vtk'




# Define project bounds
minx,miny,maxx,maxy = proj.config.bbox
model_base = proj.config.bbox_3d['base']
model_top = proj.config.bbox_3d['top']
print("bbox=",proj.config.bbox)


from LoopStructural import GeologicalModel,  __version__
from LoopStructural.visualisation import LavaVuModelViewer
from LoopStructural.visualisation.vtk_exporter import VtkExporter
from datetime import datetime
import os
from datetime import datetime
import shutil
import logging
logging.getLogger().setLevel(logging.ERROR)
import lavavu
import numpy as np  
import math
import map2loop

from map2loop.m2l_utils import save_dtm_mesh, save_dtm_ascii, save_parameters
from map2loop.map import MapUtil
import rasterio
from scipy.interpolate import RegularGridInterpolator


nowtime=datetime.now().isoformat(timespec='minutes')   
model_name='leaflet'+'_'+nowtime.replace(":","-").replace("T","-")
if (os.path.exists(vtk_path+model_name)):
    shutil.rmtree(vtk_path+model_name)
os.mkdir(vtk_path+model_name)
filename=vtk_path+model_name+'/'+'surface_name_{}.vtk'


f=open(tmp_path+'/bbox.csv','w')
f.write('minx,miny,maxx,maxy,lower,upper\n')
ostr='{},{},{},{},{},{}\n'.format(minx,miny,maxx,maxy,model_base,model_top)
f.write(ostr)
f.close()
save_dtm_mesh(dtm_path,vtk_path+model_name+'/')

foliation_params = { 'interpolatortype':'FDI',                   
                    'nelements':1e5,  # how many tetras/voxels
                   }
fault_params = {'interpolatortype':'FDI',
                    'nelements':1e5, # how many tetras/voxels
                    'points':False ,
}

import pandas as pd
orientations=pd.read_csv('../m2l_outputs/output/orientations_clean.csv')
orientations=orientations[orientations['polarity']!=0]
orientations.to_csv('../m2l_outputs/output/orientations_clean.csv')
print("filtered orientations")

save_parameters(model_name,vtk_path+model_name,proj,foliation_params,fault_params,'',map2loop.__version__,__version__)

model,m2l_data = GeologicalModel.from_map2loop_directory(proj_path,
                                                          fault_params=fault_params,
                                                          rescale=False,
                                                          foliation_params=foliation_params,
                                                         )

#model.to_file(output_path + "/model.pickle")    

try:
   model.update()
except:
  print("constraints",model['supergroup_17'].interpolator.constraints)
  quit()

surface_verts = {}

def function(xyz,tri,name):
    xyz = np.copy(xyz)
    tri = np.copy(tri)
    nanmask = np.all(~np.isnan(xyz),axis=1)
    vert_idx = np.zeros(xyz.shape[0],dtype=int) -1
    vert_idx[nanmask] = np.arange(np.sum(nanmask))
    tri[:] = vert_idx[tri]
    tri = tri[np.all(tri > -1,axis=1),:]
    xyz = xyz[nanmask,:]
    surface_verts[name] = (xyz,tri)

def mask(xyz): # for clipping strat to surface dtm
    from map2loop.map import MapUtil
    import rasterio
    import os
    dtm_map = MapUtil(proj.config.bbox_3d,dtm=rasterio.open(os.path.join(dtm_path,'dtm_rp.tif')))
    xyz=model.rescale(xyz,inplace=False)
    dtmv = dtm_map.evaluate_dtm_at_points((xyz[:,:2]))
    return xyz[:,2]<dtmv

clip_on_dtm=True
if(clip_on_dtm):
    dtm = rasterio.open(dtm_path+'/dtm_rp.tif')
    dtm_val = dtm.read(1)
    x=np.linspace(dtm.bounds[0],dtm.bounds[2],dtm_val.shape[1])
    y=np.linspace(dtm.bounds[1],dtm.bounds[3],dtm_val.shape[0])
    dtm_interpolator = RegularGridInterpolator((x,y),np.rot90(dtm_val,3))
    model.dtm = lambda xyz : dtm_interpolator(xyz[:,:2])
view = VtkExporter(model,vtk_path+model_name+'/')

view.nsteps = np.array([200,200,200])

view.nelements = 1e5
view.add_model_surfaces(function=function,filename=filename,faults=False)
view.nelements=1e6
view.add_model_surfaces(function=function,filename=filename,strati=False,displacement_cmap = 'rainbow')


for layer in surface_verts:
    f=open(vtk_path+model_name+'/'+layer.replace("_iso_0.000000","")+'.obj','w')
    vert=surface_verts[layer][0].shape[0]
    tri=surface_verts[layer][1].shape[0]
    print(layer,vert,tri)
    for v in range(0,vert):
        ostr = "v {} {} {}\n"\
            .format(surface_verts[layer][0][v][0],surface_verts[layer][0][v][1],surface_verts[layer][0][v][2])
        f.write(ostr)
    for t in range(0,tri):
        ostr = "f {} {} {}\n"\
                .format(surface_verts[layer][1][t][0]+1,surface_verts[layer][1][t][1]+1,surface_verts[layer][1][t][2]+1)
        f.write(ostr)
    f.close()    


import pandas as pd
from os import listdir
import os
from os.path import isfile, join
from pathlib import Path
import numpy as np
from geoh5py.objects import BlockModel
from geoh5py.workspace import Workspace
from geoh5py.objects import Surface

def hextofloats(h):
    '''Takes a hex rgb string (e.g. #ffffff) and returns an RGB tuple (float, float, float).'''
    return tuple(int(h[i:i + 2], 16) / 255. for i in (1, 3, 5))  # skip '#'

def geoh5_create_surface_data(obj_path_dir,colour_path,model_name):

    h5file_path = obj_path_dir+"/loop_"+model_name.replace("leaflet_","")+".geoh5"

    workspace = Workspace(h5file_path,version=2.0)
    onlyfiles = [f for f in listdir(obj_path_dir) if isfile(join(obj_path_dir, f))]
    colour_index=0
    all_sorts = pd.read_csv(os.path.join(colour_path, 'all_sorts_clean.csv'), sep=",")
    all_sorts=all_sorts.set_index('code')
    colour_map=open(obj_path_dir+'/loop_colour_map.clr','w')
    colour_map.write('{\tStart\tRed\tGreen\tBlue\t}\n')

    for file in onlyfiles:
        if ('.obj' in file):
            obj=pd.read_csv(obj_path_dir+'/'+file,sep=' ',names=["code","X","Y","Z"])
            indices=obj[obj['code']=='f']
            vertices=obj[obj['code']=='v']
            vertices=vertices.drop(['code'], axis=1)
            indices=indices[list("XYZ")].astype(int)
            i=indices.to_numpy()-1
            v=vertices.to_numpy()
            if(len(i)>0 and len(v)>0):
                # Create a geoh5 surface
                surface = Surface.create(
                          workspace, name=file.replace('.obj',''), vertices=v, cells=i
                            )
                if('Fault_' in file or 'dtm' in file):
                    colours=np.ones(surface.n_cells)*99
                else:
                    colours=np.ones(surface.n_cells)*colour_index
                    rgb=hextofloats(all_sorts.loc[file.replace('.obj','')]['colour'])
                    colour_map.write('{}\t{}\t{}\t{}\n'.format(colour_index,rgb[0],rgb[1],rgb[2]))
                    colour_index=colour_index+1
                    

                surface.add_data({
                    "colour_index": {
                        "association":"CELL",
                        "values": colours
                    }
                })
                
                workspace.save_entity(surface)
    workspace.close()
                
    colour_map.write('{}\t{}\t{}\t{}\n'.format(99,1,1,1))
    colour_map.close()
    print("colour map saved as:",obj_path_dir+'/loop_colour_map.clr')


obj_path_dir= vtk_path+model_name+'/'   # directory of existing obj surfaces
colour_path=tmp_path+'/'
geoh5_create_surface_data(obj_path_dir,colour_path,model_name)


#code to take a LoopStructural voxel model and save it out
#as a *.geoh5 GeoscienceAnalyst model 
#weird indexing because default LS block has X & Y swapped & Z -ve
#assumes model already created by LoopStructural, and minx, maxx info from main calculations
#Requires installation of https://github.com/MiraGeoscience/geoh5py 

# assumes LoopStructural model object has been calculated

from pathlib import Path
import numpy as np
from geoh5py.objects import BlockModel
from geoh5py.workspace import Workspace
 
voxel_size=100
sizex=int((maxx-minx)/voxel_size)
sizey=int((maxy-miny)/voxel_size)
sizez=int((model_top-model_base)/voxel_size)
nsteps=[sizex,sizey,sizez]

def create_geoh5_block_model_data(model,voxel_size,minx,miny,maxx,maxy,model_base,model_top,output_dir,nsteps,model_name):
    
    voxels=model.evaluate_model(model.regular_grid(nsteps=(nsteps[0],nsteps[1],nsteps[2]),shuffle=False),scale=False)
    voxels=voxels.astype(float)

    name = "MyLoopBlockModel"

    # Generate a 3D array

    nodal_x = np.arange(0,maxx-minx+1,voxel_size)
    nodal_y = np.arange(0,maxy-miny+1,voxel_size)
    nodal_z = np.arange(model_top-model_base+1,0,-voxel_size)

    h5file_path = output_dir+"/loop_"+model_name.replace("leaflet_","")+".geoh5"


    # Create a workspace
    workspace = Workspace(h5file_path,version=2.0)

    grid = BlockModel.create(
        workspace,
        origin=[minx+(voxel_size/2), miny+(voxel_size/2), model_base+(voxel_size/2)],
        u_cell_delimiters=nodal_x,
        v_cell_delimiters=nodal_y,
        z_cell_delimiters=nodal_z,
        name=name,
        rotation=0,
        allow_move=False,
    )
    data = grid.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": (
                    voxels.reshape((nodal_x.shape[0]-1,nodal_y.shape[0]-1,nodal_z.shape[0]-1)).transpose((1,0,2))
                ),
            }
        }
    )
    workspace.save_entity(grid)
    workspace.close()

output_dir= obj_path_dir   # output directory to save geoh5 format voxel mdoel
create_geoh5_block_model_data(model,voxel_size,minx,miny,maxx,maxy,model_base,model_top,output_dir,nsteps,model_name)

