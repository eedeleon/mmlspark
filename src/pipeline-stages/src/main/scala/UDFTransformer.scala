// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{ParamMap, UDFParam, UDPythonFParam}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.sql.execution.python.UserDefinedPythonFunction
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.{Column, DataFrame, Dataset}

object UDFTransformer extends ComplexParamsReadable[UDFTransformer]

/** <code>UDFTransformer</code> takes as input input column, output column, and a UserDefinedFunction
  * returns a dataframe comprised of the original columns with the output column as the result of the
  * udf applied to the input column
  */
@InternalWrapper
class UDFTransformer(val uid: String) extends Transformer with Wrappable with ComplexParamsWritable
  with HasInputCol with HasOutputCol {
  def this() = this(Identifiable.randomUID("UDFTransformer"))
  val udfScalaKey = "udfScala"
  val udfPythonKey = "udfPython"

  val udfScala = new UDFParam(this, udfScalaKey, "User Defined Function to be applied to the DF input col")
  val udfPython = new UDPythonFParam(this, udfPythonKey,
    "User Defined Python Function to be applied to the DF input col")
  val udfParams = Seq(udfScala, udfPython)

  /** @group getParam */
  def getUDF: UserDefinedFunction = $(udfScala)

  /** @group getParam */
  def getUDPythonF: UserDefinedPythonFunction = $(udfPython)

  /** @group setParam */
  def setUDF(value: UserDefinedFunction): this.type = {
    udfParams.foreach(clear)
    set(udfScalaKey, value)
  }

  /** @group setParam */
  def setUDF(value: UserDefinedPythonFunction): this.type = {
    udfParams.foreach(clear)
    set(udfPythonKey, value)
  }

  def applyUDF(col: Column): Column =  if (isSet(udfScala)) getUDF(col) else getUDPythonF(col)

  def getDataType: DataType =  if (isSet(udfScala)) getUDF.dataType else getUDPythonF.dataType

  /** @param dataset - The input dataset, to be transformed
    * @return The DataFrame that results from applying the udf to the inputted dataset
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    dataset.withColumn(getOutputCol, applyUDF(dataset.col(getInputCol)))
  }

  def validateAndTransformSchema(schema: StructType): StructType = {
    val col = schema(getInputCol)
    schema.add(StructField(getOutputCol, getDataType, col.nullable, col.metadata))
  }

  def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  def copy(extra: ParamMap): UDFTransformer = defaultCopy(extra)

}

