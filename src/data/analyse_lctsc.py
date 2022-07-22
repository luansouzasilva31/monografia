import os
import glob
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
from dcmrtstruct2nii import dcmrtstruct2nii


def extract_nii(basepath=None):
    # Importando csv e lendo informações necessárias
    rspaths = pd.read_csv(os.path.join(f'{os.sep}'.join(basepath.split(os.sep)[:-1]), 'metadata.csv'))['File Location']
    
    # Percorrendo arquivos rstruct e dcm
    for rspath in rspaths:
        # Definindo caminho para rstruct
        rspath = os.path.join(basepath, f'{os.sep}'.join(rspath.split(os.sep)[2:]))
        
        # Definindo caminho para ct scans
        ctpath = glob.glob(os.path.join(f'{os.sep}'.join(rspath.split(os.sep)[:-1]), '*'))
        ctpath.remove(rspath)
        ctpath = ctpath[0]
        
        # Definindo caminho de salvamento dos dados
        outpath = rspath.replace('datasets_originais', 'datasets').split(os.sep)
        outpath = os.path.join(f'{os.sep}'.join(outpath[:3]), outpath[4])
        
        # Salvando todos os arquivos associados ao DICOM e RT Struct
        print(f'\t{outpath.split(os.sep)[-1]}...', end=' ')
        try:
            dcmrtstruct2nii(os.path.join(rspath, '1-1.dcm'), ctpath, outpath)
            print('OK')
            
            try:
                os.remove(os.path.join(outpath, 'mask_Heart.nii.gz'))
            except:
                pass
            
            try:
                os.remove(os.path.join(outpath, 'mask_Esophagus.nii.gz'))
            except:
                pass
            
            try:
                os.remove(os.path.join(outpath, 'mask_SpinalCord.nii.gz'))
            except:
                pass
        
        except:
            print('inconsistent!')
        
        # Gerando máscara completa a partir das máscaras pulmonares direita e esquerda
        try:
            rmask = sitk.ReadImage(os.path.join(outpath, 'mask_Lung_R.nii.gz'))
            lmask = sitk.ReadImage(os.path.join(outpath, 'mask_Lung_L.nii.gz'))
            
            rmask = sitk.GetArrayFromImage(rmask)
            lmask = sitk.GetArrayFromImage(lmask)
            
            mask = np.sum((rmask, lmask), axis=0)
            
            mask = sitk.GetImageFromArray(mask)
            sitk.WriteImage(mask, os.path.join(outpath, 'mask_Lung.nii.gz'))
        
        except Exception:
            print(f'\t{os.path.basename(outpath)} doesn\'t have all the expected masks. Deleting ...', end=' ')
            shutil.rmtree(outpath, ignore_errors=True)
            print('Done.')
        
        '''Extraindo data com rt struct
        rtstruct = RTStructBuilder.create_from(dicom_series_path=ctpath,
                                               rt_struct_path=rspath)
        
        print(rtstruct.get_roi_names())
        
        Extraindo máscaras
        rmask = rtstruct.get_roi_mask_by_name('Lung_R')
        lmask = rtstruct.get_roi_mask_by_name('Lung_L')
        mask = np.sum((lmask, rmask), axis=0)
        
        Convertendo DICOM para .nii
        dicom2nifti.dicom_series_to_nifti(ctpath, './teste.nii.gz')
        
        Convertendo array para .nii
        mask_nii = array_to_nii(mask, affine=np.eye(4))
        mask_nii.to_filename('./mask.nii.gz')
        
        n = 50
        plt.imshow(mask[n, :, :])'''


def analyse_shape(basepath=None):
    # Verificando se shape é o adequado
    for fold in ['train', 'test']:
        foldpath = os.path.join(basepath, fold)
        datapaths = glob.glob(os.path.join(foldpath, '*'))
        
        for datapath in datapaths:
            # Importando ct_scan e correspondentes máscaras pulmonares
            ct_scan = sitk.ReadImage(os.path.join(datapath, 'image.nii.gz'))
            mask_left = sitk.ReadImage(os.path.join(datapath, 'mask_Lung_L.nii.gz'))
            mask_right = sitk.ReadImage(os.path.join(datapath, 'mask_Lung_R.nii.gz'))
            
            # Verificando se há amostras com shapes inconsistentes entre si:
            consistent1 = ct_scan.GetSize() == mask_left.GetSize() and \
                          ct_scan.GetSize() == mask_right.GetSize() and \
                          mask_left.GetSize() == mask_right.GetSize()
            
            # Verificando se há amostras com shapes fora do padrão esperado
            consistent2 = ct_scan.GetSize()[0] == ct_scan.GetSize()[1]
            
            consistent = consistent1 and consistent2
            
            # Deletando amostras inconsistentes
            print(
                    f'{datapath.split(os.sep)[-1]} >>> ct:{ct_scan.GetSize()}, mask_L:{mask_left.GetSize()}, '
                    f'mask_R:{mask_right.GetSize()}',
                    end=' ')
            if not consistent:
                print('inconsistent! Being removed ...', end=' ')
                shutil.rmtree(datapath, ignore_errors=True)
                print('Done!')
            
            else:
                print('Good!')


def split_data(basepath=None):
    datapaths = glob.glob(os.path.join(basepath, '*'))
    
    trainpath = os.path.join(basepath, 'train')
    testpath = os.path.join(basepath, 'test')
    
    if not os.path.exists(trainpath): os.makedirs(trainpath)
    if not os.path.exists(testpath): os.makedirs(testpath)
    
    for datapath in datapaths:
        if 'train' in datapath.lower():
            shutil.move(datapath, os.path.join(trainpath, datapath.split(os.sep)[-1]))
        
        elif 'test' in datapath.lower():
            shutil.move(datapath, os.path.join(testpath, datapath.split(os.sep)[-1]))
        
        else:
            print(f'{datapath.split(os.sep)[-1]} neither train or test. Skipping!')


def generating_metadata(basepath=None):
    print('Generating csv, please wait a few minutes...', end=' ')
    
    metadata = []
    
    for folder in ['train', 'test']:
        csv = pd.DataFrame(columns=['name', 'folder', 'ct_path', 'mask_L_path', 'mask_R_path', 'shape'])
        
        data_paths = glob.glob(os.path.join(basepath, folder, '*'))
        
        for i, data_path in enumerate(data_paths):
            # Carregando ct scan
            mask = sitk.ReadImage(os.path.join(data_path, 'mask_Lung_L.nii'))
            
            # Preenchendo CSV
            csv.at[i, 'name'] = data_path.split(os.sep)[-1]
            csv.at[i, 'folder'] = folder
            csv.at[i, 'ct_path'] = os.path.join(data_path, 'image.nii.gz')
            csv.at[i, 'mask_L_path'] = os.path.join(data_path, 'mask_Lung_L.nii.gz')
            csv.at[i, 'mask_R_path'] = os.path.join(data_path, 'mask_Lung_R.nii.gz')
            csv.at[i, 'mask_path'] = os.path.join(data_path, 'mask_Lung.nii.gz')
            csv.at[i, 'shape'] = mask.GetSize()
        
        metadata.append(csv)
    
    # Concatenando csvs gerados
    metadata = pd.concat(metadata, axis=0, sort=False, ignore_index=True)
    
    # Salvando csv
    metadata.to_csv(os.path.join(basepath, 'metadata.csv'), index=False)
    
    print('Done.')


if __name__ == '__main__':
    # Extraindo arquivos na extensão .nii
    extract_nii(basepath=os.path.join('D:', 'Monografia', 'datasets_originais', 'LCTSC', 'data'))
    
    # Aplicando split nos dados (train/test)
    split_data(basepath=os.path.join('D:', 'Monografia', 'datasets', 'LCTSC'))
    
    # Analisando dados .nii
    analyse_shape(basepath=os.path.join('D:', 'Monografia', 'datasets', 'LCTSC'))
    
    # Gerando csv
    generating_metadata(basepath=os.path.join('D:', 'Monografia', 'datasets', 'LCTSC'))
