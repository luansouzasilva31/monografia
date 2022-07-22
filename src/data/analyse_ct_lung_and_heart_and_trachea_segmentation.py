import os
import glob
import shutil
import dicom2nifti
import pandas as pd
import SimpleITK as sitk
from termcolor import colored


def extract_nii(basepath=None) :
    # Masks
    print('MASKS')
    
    path = os.path.join(basepath, 'masks', 'nrrd_lung', 'nrrd_lung')
    mask_names = os.listdir(path)
    mask_names = [mask_name.replace('_lung.nrrd', '') for mask_name in mask_names]
    
    for mask_name in mask_names :
        print(f'\t {mask_name}', end='... ')
        
        image = sitk.ReadImage(os.path.join(path, mask_name + '_lung.nrrd'))
        
        outpath = os.path.join(basepath, 'masks')
        outpath = outpath.replace('datasets_originais', 'datasets')
        if not os.path.exists(outpath) : os.makedirs(outpath)
        
        outpath = os.path.join(outpath, mask_name)
        
        sitk.WriteImage(image, outpath + '.nii.gz')
        
        print('OK')
    
    # CT Scans
    print('CT SCANS')
    
    path = os.path.join(basepath, 'ct_scans')
    for folder in ['train', 'test'] :
        print('Folder: ', folder)
        patients = glob.glob(os.path.join(path, folder, '*'))
        
        for patient in patients :
            print(f'\t {patient.split(os.sep)[-1]}', end='... ')
            
            outpath = patient.replace('datasets_originais', 'datasets')
            
            if not os.path.exists(f'{os.sep}'.join(outpath.split(os.sep)[:-1])) :
                os.makedirs(f'{os.sep}'.join(outpath.split(os.sep)[:-1]))
            
            try :
                dicom2nifti.dicom_series_to_nifti(patient, outpath + '.nii.gz')
                print('OK')
            except :
                print('inconsistent!')


def deleting_missing_data(basepath=None) :
    # Deletando "test" por serem amostras repetidas de "train"
    print('Deleting `test` folder ...', end=' ')
    try :
        shutil.rmtree(os.path.join(basepath, 'ct_scans', 'test'))
        print('Done!')
    except FileNotFoundError :
        print('Oh! The file was already deleted!')
    
    # "Extraindo" conteúdo de "train" para "ct_scans"
    shutil.move(os.path.join(basepath, 'ct_scans', 'train'), os.path.join(basepath, 'train'))
    shutil.rmtree(os.path.join(basepath, 'ct_scans'))
    os.rename(os.path.join(basepath, 'train'), os.path.join(basepath, 'ct_scans'))
    
    # Apagando dados inconsistentes ou incompletos
    ct_names = os.listdir(os.path.join(basepath, 'ct_scans'))
    mask_names = os.listdir(os.path.join(basepath, 'masks'))
    
    print('Analysing CT Scans...')
    for name in ct_names :
        if not name in mask_names :
            print(f'\t Deleting {name} ...', end=' ')
            os.remove(os.path.join(basepath, 'ct_scans', name))
            print('Done.')
    
    print('Analysing masks...')
    for name in mask_names :
        if not name in ct_names :
            print(f'\t Deleting {name} ...', end=' ')
            os.remove(os.path.join(basepath, 'masks', name))
            print('Done.')
    
    print(colored('Process finished.', 'green'))


def analyse_shape(basepath=None) :
    ct_paths = glob.glob(os.path.join(basepath, 'ct_scans', '*.nii.gz'))
    mask_paths = [ct_path.replace('ct_scans', 'masks') for ct_path in ct_paths]
    
    for ct_path, mask_path in zip(ct_paths, mask_paths) :
        # Importando ct_scan e mask
        try :
            ct_scan = sitk.ReadImage(ct_path)
            mask = sitk.ReadImage(mask_path)
            
            # Verificando se há amostras com shapes inconsistentes entre si:
            consistent1 = ct_scan.GetSize() == mask.GetSize()
            
            # Verificando se há amostras com shapes fora do padrão esperado
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
        
        except Exception :
            print(f'{ct_path.split(os.sep)[-1]} >>> ERROR! Being removed ...', end=' ')
            os.remove(ct_path)
            os.remove(mask_path)
            print('Done!')


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
    test_names = ['ID00025637202179541264076.nii.gz', 'ID00019637202178323708467.nii.gz',
                  'ID00360637202295712204040.nii.gz', 'ID00364637202296074419422.nii.gz',
                  'ID00365637202296085035729.nii.gz', 'ID00370637202296737666151.nii.gz',
                  'ID00383637202300493233675.nii.gz', 'ID00392637202302319160044.nii.gz',
                  'ID00398637202303897337979.nii.gz', 'ID00400637202305055099402.nii.gz',
                  'ID00405637202308359492977.nii.gz', 'ID00407637202308788732304.nii.gz',
                  'ID00411637202309374271828.nii.gz', 'ID00417637202310901214011.nii.gz',
                  'ID00423637202312137826377.nii.gz', 'ID00426637202313170790466.nii.gz']
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
    # Extraindo dados em .nii
    extract_nii(basepath=os.path.join('D:', 'Monografia', 'datasets_originais',
                                      'ct_lung_and_heart_and_trachea_segmentation'))
    
    # Detectando dados repetidos
    deleting_missing_data(basepath=os.path.join('D:', 'Monografia', 'datasets',
                                                'ct_lung_and_heart_and_trachea_segmentation'))
    
    # Analisando dados .nii
    analyse_shape(basepath=os.path.join('D:', 'Monografia', 'datasets', 'ct_lung_and_heart_and_trachea_segmentation'))
    
    # Aplicando split nos dados (train/test)
    split_data(basepath=os.path.join('D:', 'Monografia', 'datasets', 'ct_lung_and_heart_and_trachea_segmentation'))
    
    # Gerando csv
    generating_metadata(basepath=os.path.join('D:', 'Monografia', 'datasets',
                                              'ct_lung_and_heart_and_trachea_segmentation'))
