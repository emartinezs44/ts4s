val ScalaVersion = "3.3.0"
val ts4sVersion  = "0.0.1"

val bigdlJarPath = ""

lazy val bigDlDlibArtifact =
  bigdlJarPath + "bigdl-dllib-spark_3.2.3-2.3.0-SNAPSHOT-jar-with-dependencies.jar"
lazy val onnxRuntimeVersion        = "1.15.0"
lazy val sparkVersion              = "3.2.0"
lazy val zooCoreVersion            = "2.3.0"
lazy val bigdlVersion              = "2.3.0-SNAPSHOT"
lazy val javaCppOnnxPresetsVersion = "1.9.0-1.5.6"
lazy val bigdlDlibPackage          = "bigdl-dllib-spark_3.2.3"
lazy val scalaTestVersion          = "3.2.16"

lazy val onnxRuntime = "com.microsoft.onnxruntime" % "onnxruntime" % onnxRuntimeVersion
lazy val zooCore =
  "com.intel.analytics.zoo" % "zoo-core-dist-all" % zooCoreVersion

lazy val sparkMl = ("org.apache.spark" %% "spark-mllib" % sparkVersion)
  .cross(CrossVersion.for3Use2_13)

lazy val sparkSql =
  ("org.apache.spark" %% "spark-sql" % sparkVersion).cross(CrossVersion.for3Use2_13)

lazy val bigdlDlib =
  ("com.intel.analytics.bigdl" % bigdlDlibPackage % bigdlVersion from bigDlDlibArtifact)
    .excludeAll(ExclusionRule(organization = "org.scalactic", "scalactic"))
    .cross(CrossVersion.for3Use2_13)

lazy val javaCppPresets =
  "org.bytedeco" % "onnx-platform" % javaCppOnnxPresetsVersion

val jsoniterScalaCore = "com.github.plokhotnyuk.jsoniter-scala" %% "jsoniter-scala-core" % "2.23.3"

// Use the "provided" scope instead when the "compile-internal" scope is not supported
val jsoniterScalaMacros = "com.github.plokhotnyuk.jsoniter-scala" %% "jsoniter-scala-macros" % "2.23.3" % "provided"

val sparkScala3 = "io.github.vincenzobaz" %% "spark-scala3" % "0.2.1"

lazy val ts4s = project
  .in(file("."))
  .settings(
    name                    := "ts4s",
    organization            := "em.ml",
    scalaVersion            := ScalaVersion,
    version                 := ts4sVersion,
    ThisBuild / useCoursier := true,
    libraryDependencies ++= Seq(
      bigdlDlib,
      sparkSql,
      sparkMl,
      onnxRuntime,
      zooCore,
      javaCppPresets,
      "com.github.sbt" % "junit-interface" % "0.13.3" % Test,
      jsoniterScalaCore,
      jsoniterScalaMacros,
      sparkScala3
    ),
    excludeDependencies += "org.scala-lang.modules" % "scala-xml_2.13"
  )

resolvers += "ossrh repository" at "https://oss.sonatype.org/content/repositories/snapshots/"
