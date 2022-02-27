import os
import time
from sshtunnel import SSHTunnelForwarder
import paramiko
import pymysql
import pandas as pd
"""
Example:
dbParam = {
    'sql_hostname' : 'ca-production',
    'sql_username' : 'name',
    'sql_password' : 'password',
    'sql_main_database' : 'ca_production',
    'sql_port' : 0000,
    'ssh_host' : '00.00.000.000',
    'ssh_user' : 'deploy',
    'ssh_port' : 22,
    'local_port': 33306
}

getDBData(dbParam, your_query)


"""


def getDBData(dbParam, query, connection_buffer=1, connect_timeout=1000, use_own_local_port=False):
    '''
    Parameters
    ----------
    db: which db to retrieve data from
    query: consist of the MYSQL query to run on the dbs
    connection_buffer: time(sec) give for connecting to MySQL, if query before the
    connection is made, Timeout error will show
    Returns
    -------
    df: dataframe that contains the query results
    '''

    home = os.path.expanduser('~')
    # set up your own info, if problem when open pkeyfile, learn from
    # https://serverfault.com/questions/939909/ssh-keygen-does-not-create-rsa-private-key
    pkeyfilepath = '/.ssh/id_rsa'
    my_pubkey = paramiko.RSAKey.from_private_key_file(home + pkeyfilepath)

    with SSHTunnelForwarder(
        (dbParam['ssh_host'], dbParam['ssh_port']), ssh_username=dbParam['ssh_user'],
            ssh_pkey=my_pubkey,
            remote_bind_address=(dbParam['sql_hostname'], dbParam['sql_port'])) as tunnel:
        if use_own_local_port:
            conn = pymysql.connect(
                host='127.0.0.1',
                user=dbParam['sql_username'],
                passwd=dbParam['sql_password'],
                db=dbParam['sql_main_database'],
                port=dbParam['local_port'],
                connect_timeout=connect_timeout
            )
            time.sleep(connection_buffer)
            df = pd.read_sql_query(query, conn)
        else:
            conn = pymysql.connect(
                host='127.0.0.1',
                user=dbParam['sql_username'],
                passwd=dbParam['sql_password'],
                db=dbParam['sql_main_database'],
                port=tunnel.local_bind_port,
                connect_timeout=connect_timeout
            )
            time.sleep(connection_buffer)
            df = pd.read_sql_query(query, conn)

        conn.close()
    return df


def create(dbParam, query):
    """
    Create example

        create_table='''
    CREATE TABLE recipes (
      recipe_id INT NOT NULL,
      recipe_name VARCHAR(30) NOT NULL,
      PRIMARY KEY (recipe_id),
      UNIQUE (recipe_name)
    );
    '''

    insert_info='''
    INSERT INTO recipes
        (recipe_id, recipe_name)
    VALUES
        (1,"Tacos"),
        (2,"Tomato Soup"),
        (3,"Grilled Cheese");
    '''

    create('test_datawarehouse',create_table)
    create('test_datawarehouse',insert_info)

    :param db: see above example
    :param query: see above example
    :return:
    """

    home = os.path.expanduser('~')
    # set up your own info, if problem when open pkeyfile, learn from
    # https://serverfault.com/questions/939909/ssh-keygen-does-not-create-rsa-private-key
    pkeyfilepath = '/.ssh/id_rsa'
    my_pubkey = paramiko.RSAKey.from_private_key_file(home + pkeyfilepath)

    with SSHTunnelForwarder(
        (dbParam['ssh_host'], dbParam['ssh_port']), ssh_username=dbParam['ssh_user'],
            ssh_pkey=my_pubkey,
            remote_bind_address=(dbParam['sql_hostname'], dbParam['sql_port'])) as tunnel:
        conn = pymysql.connect(
            host='127.0.0.1',
            user=dbParam['sql_username'],
            passwd=dbParam['sql_password'],
            db=dbParam['sql_main_database'],
            port=tunnel.local_bind_port
        )
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        conn.close()


def insert_df(dbParam, insert_table_name, df, columns=None, top_rows=None):
    """
    This function can insert a dataframe data into a existing table in db
    Parameters
    ----------
    dbParam: the credentials
    insert_table_name: which table in the db we want to insert to
    df: the df we want to insert
    columns: list or None, we can select which columns of the dataframe need to be inserted, this can also be done to the df before call this function
    top_rows: int or None, we can select top number of rows to insert, not the whole dataframe

    Returns
    -------

    Examples:
    create_table='''
    CREATE TABLE recipes (
      recipe_id INT NOT NULL,
      recipe_name VARCHAR(30) NOT NULL,
      PRIMARY KEY (recipe_id),
      UNIQUE (recipe_name)
    );
    '''
    df = pd.DataFrame({'recipe_id':[1, 2, 3],'recipe_name':['Tacos', 'Tomato Soup', 'Grilled Cheese']})


    create('test_datawarehouse',create_table)
    insert_df(dbParam, insert_table_name='recipes', df=df, columns=None, top_rows=None)

    """
    data = df.copy()
    if columns:
        data = data[columns]
    else:
        pass
    if top_rows:
        pass
    else:
        data = data.head(top_rows)

    home = os.path.expanduser('~')
    # set up your own info, if problem when open pkeyfile, learn from
    # https://serverfault.com/questions/939909/ssh-keygen-does-not-create-rsa-private-key
    pkeyfilepath = '/.ssh/id_rsa'
    my_pubkey = paramiko.RSAKey.from_private_key_file(home + pkeyfilepath)

    with SSHTunnelForwarder(
        (dbParam['ssh_host'], dbParam['ssh_port']), ssh_username=dbParam['ssh_user'],
            ssh_pkey=my_pubkey,
            remote_bind_address=(dbParam['sql_hostname'], dbParam['sql_port'])) as tunnel:
        conn = pymysql.connect(
            host='127.0.0.1',
            user=dbParam['sql_username'],
            passwd=dbParam['sql_password'],
            db=dbParam['sql_main_database'],
            port=tunnel.local_bind_port
        )
        cursor = conn.cursor()
        cols = "`,`".join([str(i) for i in data.columns.tolist()])

        # Insert DataFrame records one by one.
        for i, row in data.iterrows():
            sql = f"INSERT INTO `{insert_table_name}` (`" + cols + "`) VALUES (" + "%s," * (
                len(row) - 1
            ) + "%s)"
            cursor.execute(sql, tuple(row))

            # the connection is not autocommitted by default, so we must commit to save our changes
            conn.commit()
        conn.close()
