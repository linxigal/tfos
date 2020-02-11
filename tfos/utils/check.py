#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/15 9:32
:File   :check.py
:content:
  
"""

import functools
import inspect


def auto_type_checker(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # fetch the argument name list.
        parameters = inspect.signature(function).parameters
        argument_list = list(parameters.keys())

        # fetch the argument checker list.
        checker_list = [parameters[argument].annotation for argument in argument_list]

        # fetch the value list.
        value_list = [inspect.getcallargs(function, *args, **kwargs)[argument] for argument in
                      inspect.getfullargspec(function).args]

        # initialize the result dictionary, where key is argument, value is the checker result.
        result_dictionary = dict()
        for argument, value, checker in zip(argument_list, value_list, checker_list):
            result_dictionary[argument] = check(argument, value, checker, function)

        # fetch the invalid argument list.
        invalid_argument_list = [key for key in argument_list if not result_dictionary[key]]

        # if there are invalid arguments, raise the error.
        if len(invalid_argument_list) > 0:
            raise Exception(invalid_argument_list)

        # check the result.
        result = function(*args, **kwargs)
        checker = inspect.signature(function).return_annotation
        if not check('return', result, checker, function):
            raise Exception(['return'])

        # return the result.
        return result

    return wrapper


def check(name, value, checker, function):
    if isinstance(checker, (tuple, list, set)):
        return True in [check(name, value, sub_checker, function) for sub_checker in checker]
    elif checker is inspect._empty:
        return True
    elif checker is None:
        return value is None
    elif isinstance(checker, type):
        return isinstance(value, checker)
    elif callable(checker):
        result = checker(value)
        return result
