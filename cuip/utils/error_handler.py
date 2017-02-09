def exception(logger):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur
 
    Parameters
    ----------
    logger: logging object
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                # log the exception
                err_msg = "Exception in "+\
                    str(func.__name__)
                logger.warning(err_msg+" : {msg}".\
                                   format(msg=ex))
            raise
         return wrapper
    return decorator
