import os
import glob
import shutil
import pandas as pd
import SimpleITK as sitk


def extract_nii(basepath=None) :
    # Convertendo demais amostras para .nii
    for folder in ['ct_scans', 'masks'] :
        print('Folder: ', folder)
        inpath = os.path.join(basepath, folder, '*.mhd')
        
        for filepath in glob.glob(inpath) :
            print(f'\t {filepath.split(os.sep)[-1]}', end='... ')
            
            image = sitk.ReadImage(filepath)
            
            outpath = filepath.replace('datasets_originais', 'datasets')
            outpath = os.path.splitext(outpath)[0]
            
            if not os.path.exists(f'{os.sep}'.join(outpath.split(os.sep)[:-1])) :
                os.makedirs(f'{os.sep}'.join(outpath.split(os.sep)[:-1]))
            
            try :
                sitk.WriteImage(image, outpath + '.nii.gz')
                print('OK')
            except :
                print('inconsistent!')


def analyse_shape(basepath=None) :
    ct_paths = glob.glob(os.path.join(basepath, 'ct_scans', '*.nii.gz'))
    mask_paths = [ct_path.replace('ct_scans', 'masks') for ct_path in ct_paths]
    
    for ct_path, mask_path in zip(ct_paths, mask_paths) :
        # Importando ct_scan e mask
        ct_scan = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)
        
        # Verificando se há amostras inconsistentes
        consistent1 = ct_scan.GetSize() == mask.GetSize()
        consistent2 = ct_scan.GetSize()[0] == ct_scan.GetSize()[1]
        
        consistent = consistent1 and consistent2
        
        # Deletando amostras inconsistentes
        print(f'{ct_path.split(os.sep)[-1]} >>> ct:{ct_scan.GetSize()}, mask:{mask.GetSize()}', end=' ')
        if not consistent :
            print('inconsistent! Being removed ...', end=' ')
            os.remove(ct_path)
            os.remove(mask_path)
            print('Done!')
        
        else :
            print('Good!')


def split_data(basepath=None) :
    # Definindo e criando paths necessários
    trainpath = os.path.join(basepath, 'train')
    testpath = os.path.join(basepath, 'test')
    
    if not os.path.exists(trainpath) :
        os.makedirs(os.path.join(trainpath, 'ct_scans'))
        os.makedirs(os.path.join(trainpath, 'masks'))
    if not os.path.exists(testpath) :
        os.makedirs(os.path.join(testpath, 'ct_scans'))
        os.makedirs(os.path.join(testpath, 'masks'))
    
    # Movendo test data
    test_names = ['VESSEL12_01.nii.gz', 'VESSEL12_10.nii.gz', 'VESSEL12_20.nii.gz']
    for name in test_names :
        shutil.move(os.path.join(basepath, 'ct_scans', name),
                    os.path.join(testpath, 'ct_scans', name))
        shutil.move(os.path.join(basepath, 'masks', name),
                    os.path.join(testpath, 'masks', name))
    
    # Movendo train data
    names = os.listdir(os.path.join(basepath, 'ct_scans'))
    
    for name in names :
        shutil.move(os.path.join(basepath, 'ct_scans', name),
                    os.path.join(trainpath, 'ct_scans', name))
        shutil.move(os.path.join(basepath, 'masks', name),
                    os.path.join(trainpath, 'masks', name))
    
    # Deletando pastas não mais necessárias :')
    shutil.rmtree(os.path.join(basepath, 'ct_scans'))
    shutil.rmtree(os.path.join(basepath, 'masks'))


def generating_metadata(basepath=None) :
    print('Generating csv, please wait a few minutes...', end=' ')
    
    metadata = []
    
    for folder in ['train', 'test'] :
        csv = pd.DataFrame(columns=['name', 'folder', 'ct_path', 'mask_path', 'shape'])
        
        mask_paths = glob.glob(os.path.join(basepath, folder, 'masks', '*.nii.gz'))
        
        for i, mask_path in enumerate(mask_paths) :
            # Carregando mask para ver shape
            mask = sitk.ReadImage(mask_path)
            
            # Preenchendo CSV
            csv.at[i, 'name'] = mask_path.split(os.sep)[-1]
            csv.at[i, 'folder'] = folder
            csv.at[i, 'ct_path'] = mask_path.replace('masks', 'ct_scans')
            csv.at[i, 'mask_path'] = mask_path
            csv.at[i, 'shape'] = mask.GetSize()
        
        metadata.append(csv)
    
    # Concatenando csvs gerados
    metadata = pd.concat(metadata, axis=0, sort=False, ignore_index=True)
    
    # Salvando csv
    metadata.to_csv(os.path.join(basepath, 'metadata.csv'), index=False)
    
    print('Done.')


if __name__ == '__main__' :
    # Extraindo arquivos na extensão ..ni.gz
    extract_nii(basepath=os.path.join('D:', 'Monografia', 'datasets_originais', 'lung_vessel_segmentation'))
    
    # Analisando dados .nii
    analyse_shape(basepath=os.path.join('D:', 'Monografia', 'datasets', 'lung_vessel_segmentation'))
    
    # Aplicando split nos dados (train/test)
    split_data(basepath=os.path.join('D:', 'Monografia', 'datasets', 'lung_vessel_segmentation'))
    
    # Gerando csv
    generating_metadata(basepath=os.path.join('D:', 'Monografia', 'datasets', 'lung_vessel_segmentation'))
