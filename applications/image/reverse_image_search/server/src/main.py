import uvicorn
import os
from diskcache import Cache
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.param_functions import Form
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from encode import Resnet50
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from config import TOP_K, UPLOAD_PATH
from operations.load import do_load
from operations.upload import do_upload
from operations.search import do_search
from operations.count import do_count
from operations.drop import do_drop
from operations.delete_by_id import do_delete_by_id
from operations.get_all import do_get_all
from logs import LOGGER
from pydantic import BaseModel
from typing import Optional, List, Dict
from urllib.request import urlretrieve
from fastapi import HTTPException

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL = Resnet50()
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()

# Mkdir '/tmp/search-images',系统默认存储上传图片的位置
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info(f"mkdir the path:{UPLOAD_PATH}")


# 用于直接取得图源中的图片：前端参照该路径：http://127.0.0.1:5000/data?image_path=tmp/search-images
@app.get('/data')
def get_img(image_path):
    # Get the image file
    try:
        LOGGER.info(f"Successfully load image: {image_path}")
        return FileResponse(image_path)
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


@app.get('/progress')
def get_progress(cache_path: str = './tmp'):
    # Get the progress of dealing with images
    try:
        cache = Cache(cache_path)
        progress = {
            "current": cache.get('current', 0),
            "total": cache.get('total', 0)
        }
        return progress
    except Exception as e:
        LOGGER.error(f"Get progress error: {e}")
        return {'status': False, 'msg': str(e)}, 400


class Item(BaseModel):
    Table: str  # 修改为必需参数
    File: str


class ImageData(BaseModel):
    milvus_id: str
    image_path: str


class PaginatedResponse(BaseModel):
    total: int
    total_pages: int
    current_page: int
    page_size: int
    data: List[ImageData]


# 此接口用于将部门图源全部转为图源特征存入向量数据库部门表中,item为图源路径
@app.post('/img/load')
async def load_images(item: Item):
    # Insert all the image under the file path to Milvus/MySQL
    try:
        if not item.Table:
            return {'status': False, 'msg': 'Table name is required'}, 400
            
        total_num = do_load(item.Table, item.File, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info(f"Successfully loaded data to table {item.Table}, total count: {total_num}")
        return "Successfully loaded data!"
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': str(e)}, 400


# 这里需要传入向量数据库部门表名以及部门图源路径（格式参照：tmp/search-images）
# 1.将图片特征存入向量数据库部门表 2.将图片地址存入mysql数据库
# url为图片的网络地址,new_path为部门图源路径
@app.post('/img/upload')
async def upload_images(image: UploadFile = File(None), url: str = None, table_name: str = None, new_path: str = None):
    # Insert the upload image to Milvus/MySQL

    try:
        # 如果部门图源路径存在则不会上传到默认目录
        if new_path is not None:
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                LOGGER.info(f"mkdir the path:{new_path}")
            if image is not None:
                content = await image.read()
                img_path = os.path.join(new_path, image.filename)
                with open(img_path, "wb+") as f:
                    f.write(content)
            elif url is not None:
                img_path = os.path.join(new_path, os.path.basename(url))
                urlretrieve(url, img_path)
            else:
                return {'status': False, 'msg': 'Image and url are required'}, 400
            vector_id = do_upload(table_name, img_path, MODEL, MILVUS_CLI, MYSQL_CLI)
            LOGGER.info(f"Successfully uploaded data, vector id: {vector_id}")
            return "Successfully loaded data: " + str(vector_id)
        else:
            # Save the upload image to server.
            if image is not None:
                content = await image.read()
                img_path = os.path.join(UPLOAD_PATH, image.filename)
                with open(img_path, "wb+") as f:
                    f.write(content)
            elif url is not None:
                img_path = os.path.join(UPLOAD_PATH, os.path.basename(url))
                urlretrieve(url, img_path)
            else:
                return {'status': False, 'msg': 'Image and url are required'}, 400
            vector_id = do_upload(table_name, img_path, MODEL, MILVUS_CLI, MYSQL_CLI)
            LOGGER.info(f"Successfully uploaded data, vector id: {vector_id}")
            return "Successfully loaded data: " + str(vector_id)
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


# 图搜 需要传入向量数据库部门表名
# str为传入的网络图片地址
@app.post('/img/search')
async def search_images(
    image: UploadFile = File(...), 
    topk: int = Form(TOP_K), 
    table_name: str = Form(..., description="部门向量数据库表名")
):
    try:
        content = await image.read()
        img_path = os.path.join(UPLOAD_PATH, image.filename)
        with open(img_path, "wb+") as f:
            f.write(content)
        paths, distances = do_search(table_name, img_path, topk, MODEL, MILVUS_CLI, MYSQL_CLI)
        res = dict(zip(paths, distances))
        res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.info(f"Successfully searched similar images in table {table_name}!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': str(e)}, 400


@app.post('/img/count')
async def count_images(table_name: str = None):
    # Returns the total number of images in the system
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of images!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


# 用于删除指定部门的向量数据库表，同时删除对应的数据库表，名字相同
@app.post('/img/drop')
async def drop_tables(table_name: str = Query(..., description="部门向量数据库表名")):
    try:
        if not table_name:
            return {'status': False, 'msg': 'Table name is required'}, 400
            
        status = do_drop(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info(f"Successfully dropped table {table_name} in Milvus and MySQL!")
        return status
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': str(e)}, 400


# 删除指定表中的指定milvus_id数据
@app.delete('/img/delete/{table_name}/{milvus_id}')
async def delete_by_id(table_name: str, milvus_id: int):
    print(f"将上传的信息 {table_name} #### {milvus_id}")
    try:
        # 删除 MySQL 中的数据
        status = do_delete_by_id(table_name, milvus_id, MILVUS_CLI, MYSQL_CLI)

        if status:
            # 获取图片路径
            image_path = MYSQL_CLI.get_image_path_by_id(table_name, str(milvus_id))

            if image_path:
                # 删除本地数据（文件）
                if os.path.exists(image_path):
                    os.remove(image_path)
                    LOGGER.info(f"Successfully deleted local data (image) at {image_path}")
                else:
                    LOGGER.warning(f"Local file at {image_path} does not exist")

            LOGGER.info(f"Successfully deleted data with id:{milvus_id} from table:{table_name}")
            return {"status": True, "msg": "Successfully deleted data"}
        else:
            return {"status": False, "msg": "Failed to delete data"}
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# 获取指定表的所有数据（分页）
@app.get('/img/all/{table_name}', response_model=PaginatedResponse)
async def get_all_data(
    table_name: str,
    page: int = Query(1, ge=1, description="页码，从1开始"),
    size: int = Query(10, ge=1, le=100, description="每页数量，1-100之间")
):
    try:
        data = do_get_all(table_name, MYSQL_CLI, page, size)
        LOGGER.info(f"Successfully got data from table:{table_name}, page:{page}, size:{size}")
        return data
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': str(e)}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000)
