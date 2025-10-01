from unitxt import register_local_catalog
from unitxt.catalog import get_from_catalog, get_local_catalogs_paths
from unitxt.artifact import UnitxtArtifactNotFoundError

LOCAL_CATALOG_PATH="local_catalog"

register_local_catalog(LOCAL_CATALOG_PATH)

def is_artifact_in_catalog(artifact_name: str, catalog_path: str=LOCAL_CATALOG_PATH):
    try:
        _ = get_from_catalog("metrics.llm_as_judge.direct.rits.phi_4", catalog_path=catalog_path)
    except UnitxtArtifactNotFoundError:
        return False
    return True

def get_system_catalog_path() -> str:
    paths = get_local_catalogs_paths()
    print(paths)

    return paths[-1]

get_system_catalog_path()
