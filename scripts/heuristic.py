def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes
def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    allowed template fields - follow python string module:
    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    anat = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T1w')
    func = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_acq-bold_bold')
    func_RL = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_acq-RL_bold')
    func_PA = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_acq-PA_bold')
    func_PARepeat = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_acq-PARepeat_bold')
    func_ep2d = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_acq-ep2d_bold')
    func_ep2dRepeat = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_acq-ep2dRepeat_bold')

    info = {anat: [], func_RL: [], func_ep2d: [], func_PA: [], func_ep2dRepeat: [],func_PARepeat: [],  func:[]} #func_LRsplit1: [], func_RLsplit1: [],
    anat_seriesdesc = ('3D T1','3D T1-weighted_ND','3D T1 MPRAGE','3D T1 _weighted','3D T1-WEIGHTED','3D T1-weighted','3D T1-weighted_ND','3D-T1-weighted_SAGITAL','3DT1weighted',' MPRAGE','3D T1 MPRAGE','3D_T1-weighted','SAG 3D T1','SAG 3D T1 FSPGR','SAG 3D T1-weighted','Sagittal 3D T1W', 'MPRAGE - Sag','MPRAGE','MPRAGE SAG IPAT ISO', 'T1','T1-weighted, 3D VOLUMETRIC', '3D T1 - weighted', 'SAG 3D T1', '3D T1-Weighted', '3D T1-WEIGHTED')
    search_terms_func = ('epi', 'epialt','*epfid2d1_66', '*epfid2d1_64')  # Add any additional terms here if needed
    funcRL_seriesdesc=('rsfMRI R-L', 'rsfMRI R>L','rsfMRI RL (FS-L)','rsfMRI_RL','rsfMRI_RL (no music, awake, eyes open)','Resting State fMRI R>L', 'Resting State fMRI R>L_ ACPC','rsfMRI_RL (Eyes Open)  R>>L','RSFMRI_RL')
    funcPA_seriesdesc=('rsfMRI_PA_ACPC','rsfMRI_PA','rsfMRI_PA___no_music,_awake,_eyes_open_')
    funcEP_seriesdesc=('rsFMRI_ep2d', 'rsFMRI ep2d','rsfMRI_ep2d', 'rsfMRI ep2d')

    for idx, s in enumerate(seqinfo):
        # Check for anatomical terms in the sequence name
        if any(term in s.series_description for term in anat_seriesdesc): #{'efgre3d', '*tfl3d1_16ns'}{'3D-T1-weighted_SAGITAL', 'T1-weighted, 3D VOLUMETRIC', '3D T1-weighted', '3D_T1-weighted', '3D-T1-weighted', '3D T1-weighted_ND'}

            info[anat].append(s.series_id)
        
        # Check for functional terms in the sequence name and specific sequence description for rsfMRI
        if any(term in s.sequence_name for term in search_terms_func) and 'rsfMRI'==s.series_description:
            info[func].append(s.series_id)
        
        # Additional checks for specific rsfMRI variants
        if ( any(term in s.series_description for term in funcRL_seriesdesc) and 'split' not in s.series_description):  #add 'rsfMRI R-L' & 'rsfMRI RL (FS-L)'                                 #rsfMRI R-L
            info[func_RL].append(s.series_id)

        if ( any(term in s.series_description for term in funcEP_seriesdesc) and 'split' not in s.series_description):       
            info[func_ep2d].append(s.series_id)

        if ( any(term in s.series_description for term in funcPA_seriesdesc) and 'split' not in s.series_description):          
            info[func_PA].append(s.series_id)

        # if 'rsfMRI_PA_repeat' in s.series_description :          #
        #     info[func_PARepeat].append(s.series_id)

        # if 'rsFMRI ep2dREPEAT' in s.series_description :          #
        #     info[func_ep2dRepeat].append(s.series_id)
        
    return info
