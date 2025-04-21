from logs import LOGGER

def do_get_all(table_name, mysql_cli, page_num=1, page_size=10):
    try:
        if not table_name:
            raise Exception("Table name is required!")
        
        if page_num < 1:
            raise Exception("Page number must be greater than 0!")
            
        if page_size < 1:
            raise Exception("Page size must be greater than 0!")
            
        data = mysql_cli.get_all_data(table_name, page_num, page_size)
        return data
    except Exception as e:
        LOGGER.error(f"Error with get all data: {e}")
        raise e 