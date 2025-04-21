import os
from logs import LOGGER

def do_delete_by_id(table_name, milvus_id, milvus_cli, mysql_cli):
    try:
        if not table_name:
            raise Exception("Table name is required!")
        
        # 直接通过 ID 获取记录
        image_path = mysql_cli.get_image_path_by_id(table_name, str(milvus_id))
        
        if not image_path:
            raise Exception(f"No data found with milvus_id: {milvus_id}")
            
        # 删除数据库记录
        status_mysql = mysql_cli.delete_by_milvus_id(table_name, milvus_id)
        status_milvus = milvus_cli.delete_entity_by_id(table_name, milvus_id)
        
        # 删除图片文件
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                LOGGER.debug(f"Successfully deleted image file: {image_path}")
            except Exception as e:
                LOGGER.error(f"Failed to delete image file: {e}")
                # 即使图片删除失败，只要数据库删除成功就返回True
        
        if status_mysql and status_milvus:
            return True
        else:
            raise Exception("Failed to delete data from databases")
    except Exception as e:
        LOGGER.error(f"Error with delete by id: {e}")
        raise e 