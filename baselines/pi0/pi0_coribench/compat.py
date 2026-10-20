"""Process-local compatibility; top-level classes support spawned data workers."""
import functools,json
from datasets import Features
from openpi.training import data_loader
ld=data_loader.lerobot_dataset
OriginalMetadata=ld.LeRobotDatasetMetadata
OriginalDataset=ld.LeRobotDataset
DATASET_ROOT=None

class LocalMetadata(OriginalMetadata):
    def __init__(self,repo_id,root=None,*pos,**kwargs):
        super().__init__(repo_id,DATASET_ROOT if DATASET_ROOT is not None else root,*pos,**kwargs)

class LocalDataset(OriginalDataset):
    def __init__(self,repo_id,root=None,*pos,**kwargs):
        kwargs['video_backend']='pyav'
        super().__init__(repo_id,DATASET_ROOT if DATASET_ROOT is not None else root,*pos,**kwargs)

def install(dataset):
    global DATASET_ROOT
    DATASET_ROOT=dataset
    original=Features.from_arrow_schema.__func__
    def compatible(cls,schema):
        metadata=dict(schema.metadata or {})
        if b'huggingface' in metadata:
            obj=json.loads(metadata[b'huggingface'])
            def visit(x):
                if isinstance(x,dict):
                    if x.get('_type')=='List': x['_type']='Sequence'
                    for value in x.values(): visit(value)
                elif isinstance(x,list):
                    for value in x: visit(value)
            visit(obj)
            metadata[b'huggingface']=json.dumps(obj).encode()
            schema=schema.with_metadata(metadata)
        return original(cls,schema)
    Features.from_arrow_schema=classmethod(compatible)
    ld.LeRobotDatasetMetadata=LocalMetadata
    ld.LeRobotDataset=LocalDataset
