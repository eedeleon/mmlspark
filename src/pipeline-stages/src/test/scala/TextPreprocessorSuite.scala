// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.util.MLReadable

class TextPreprocessorSuite extends TestBase with TransformerFuzzing[TextPreprocessor] {
  val toMap1 = "The happy sad boy drank sap"
  val toMap2 = "The hater sad doy drank sap"
  val toMap3 = "The hater sad doy"
  val wordDF = session.createDataFrame(Seq(
    (0, toMap1, "The sad sap boy drank sap"),
    (1, toMap2, "The sap sap drank sap"),
    (2, "foo", "foo"),
    (3, toMap3 + "aABc0123456789Zz_", "The sap sap")))
    .toDF("label", "words1", "words2")

  val testMap = Map[String, String] (
    "happy"   -> "sad",
    "hater"   -> "sap",
    "sad"     -> "sap",
    "sad doy" -> "sap"
  )

  val testTrie1 = new Trie().put("happy", "sad").put("hater", "sap")
  var testTrie1Pivot: Trie = testTrie1
  for (letter <- "ha") testTrie1Pivot = testTrie1Pivot.get(letter).get

  test("Check basic string mapper") {
    {
      var copyHappy: Trie = testTrie1
      for (letter <- "happy") copyHappy = copyHappy.get(letter).get
      assert(copyHappy != null)
      assert(copyHappy.value.mkString("") == "sad")
    }

    {
      var copyHater: Trie = testTrie1
      for (letter <- "hater") copyHater = copyHater.get(letter).get
      assert(copyHater != null)
      assert(copyHater.value.mkString("") == "sap")
    }

    {
      var copyHaHater: Trie = testTrie1Pivot
      for (letter <- "ppy") copyHaHater = copyHaHater.get(letter).get
      assert(copyHaHater != null)
      assert(copyHaHater.value.mkString("") == "sad")
    }

    {
      var copyHaHappy: Trie = testTrie1Pivot
      for (letter <- "ter") copyHaHappy = copyHaHappy.get(letter).get
      assert(copyHaHappy != null)
      assert(copyHaHappy.value.mkString("") == "sap")
    }
  }

  test("Check trie put vs putAll equality") {
    val test2 = new Trie().putAll(Map[String,String]("happy" -> "sad", "hater" -> "sap"))
    val testString = "happy hater"
    assert(testTrie1.mapText(testString).equals(test2.mapText(testString)))
  }

  test("Check trie text mapper") {
    val mappings = Map[String, String]("happy" -> "sad", "hater" -> "sap", "sad" -> "sap", "sad doy" -> "sap")
    val test = new Trie().putAll(mappings)
    print(test.mapText(toMap1))
    assert(test.mapText(toMap1).equals("The sad sap boy drank sap"))
    assert(test.mapText(toMap2).equals("The sap sap drank sap"))
  }

  test("Check trie text normalizer") {
    var test = new Trie(normFunction = Character.toUpperCase)
    test = test.put("happy", "sad")
    test = test.put("hater", "sap")
    test = test.put("sad", "sap")
    test = test.put("sad doy", "sap")
    test = test.put("the", "sat")
    val item = test.mapText(toMap1)
    val item1 = test.mapText(toMap2)
    assert(item.equals("sat sad sap boy drank sap"))
    assert(item1.equals("sat sap sap drank sap"))
  }
  test("Check TextPreprocessor text normalizers") {
    new TextPreprocessor()
      .setMap(testMap)
      .setInputCol("words1")
      .setOutputCol("out")
    new TextPreprocessor()
      .setMap(testMap).setInputCol("words1")
      .setOutputCol("out")
      .setNormFunc("identity")
    new TextPreprocessor()
      .setMap(testMap)
      .setInputCol("words1")
      .setOutputCol("out")
      .setNormFunc("lowerCase")
    var errorThrown = false
    try {
      new TextPreprocessor()
        .setMap(testMap)
        .setNormFunc("p")
        .setInputCol("words1")
        .setOutputCol("out")
    } catch {
      case _: Exception => errorThrown = true
    }
    assert(errorThrown)
  }

  test("Check trie text df") {
    val textPreprocessor = new TextPreprocessor()
      .setNormFunc("lowerCase")
      .setMap(testMap)
      .setInputCol("words1")
      .setOutputCol("out")
    val processed = textPreprocessor.transform(wordDF)
    val out = processed.select("out").collect()
    val ans = processed.select("words2").collect()

    assert(out(0).equals(ans(0)))
    assert(out(1).equals(ans(1)))
    assert(out(2).equals(ans(2)))
    assert(out(3).equals(ans(3)))
  }

  def testObjects(): Seq[TestObject[TextPreprocessor]] = List(new TestObject(
    new TextPreprocessor().setInputCol("words").setOutputCol("out"), makeBasicDF()))

  override def reader: MLReadable[_] = TextPreprocessor
}
