// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

object RenameColumn extends DefaultParamsReadable[RenameColumn]

/** <code>RenameColumn</code> takes a dataframe with an input and an output column name
  * and returns a dataframe comprised of the original columns with the input column renamed
  * as the output column name.
  */

class RenameColumn(val uid: String) extends Transformer with MMLParams
  with HasInputCol with HasOutputCol {
  def this() = this(Identifiable.randomUID("RenameColumn"))

  /** @param dataset - The input dataset, to be transformed
    * @return The DataFrame that results from rename the input column
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    verifyInputSchema(dataset.schema)
    dataset.toDF().withColumnRenamed(getInputCol, getOutputCol)
  }

  def transformSchema(schema: StructType): StructType = {
    verifyInputSchema(schema)
    val col = schema(getInputCol)
    val outSchema = schema.add(new StructField(getOutputCol, col.dataType, col.nullable, col.metadata))
    verifiedOutputSchema(StructType(outSchema.filter(f => !f.name.equals(getInputCol))))
  }

  def copy(extra: ParamMap): RenameColumn = defaultCopy(extra)

  private def verifyInputSchema(schema: StructType): Unit = {
    val hasInputCol = schema.fields.map(_.name).toSet.contains(getInputCol)
    if (!hasInputCol) {
      throw new Exception (s"DataFrame does not contain specified input column: $getInputCol")
    }
  }
  private def verifiedOutputSchema(schema: StructType): StructType = {
    val hasOutputCol = schema.fields.map(_.name).toSet.contains(getOutputCol)
    if (!hasOutputCol) {
      throw new Exception(s"DataFrame does not contain specified output column: $getOutputCol")
    }
    schema
  }
}
