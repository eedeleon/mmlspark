# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys

if sys.version >= '3':
    basestring = str

from mmlspark._UDFTransformer import _UDFTransformer
from pyspark.ml.common import inherit_doc
from pyspark.sql.functions import UserDefinedFunction

@inherit_doc
class UDFTransformer(_UDFTransformer):
    """

    Args:
        udf (UserDefinedFunction): The udf to be applied.
    """
    def setUDF(self, udf):
        func = None if not hasattr(udf, "func") else udf.func
        returnType = None if not hasattr(udf, "returnType") else udf.returnType
        userDefinedFunction = UserDefinedFunction(func, returnType = returnType,
                                name = __name__)
        self._java_obj = self._java_obj.setUDPythonF(userDefinedFunction._judf)
        return self


