val ScalaVersion = "3.3.0"
val ts4sVersion  = "0.0.1"

val bigdlJarPath = "file://///Users/e049627/Development/bigdl_scala_2.13_mainstream/BigDL/scala/dllib/target/"

val bigDlDlibArtifact =
  bigdlJarPath + "bigdl-dllib-spark_3.2.3-2.3.0-SNAPSHOT-jar-with-dependencies.jar"

val onnxRuntimeVersion        = "1.15.0"
val sparkVersion              = "3.2.0"
val zooCoreVersion            = "2.3.0"
val bigdlVersion              = "2.3.0-SNAPSHOT"
val javaCppOnnxPresetsVersion = "1.9.0-1.5.6"
val bigdlDlibPackage          = "bigdl-dllib-spark_3.2.3"
val scalaTestVersion          = "3.2.16"
val scoptVersion              = "4.1.0"

val onnxRuntime = "com.microsoft.onnxruntime" % "onnxruntime" % onnxRuntimeVersion
val zooCore =
  "com.intel.analytics.zoo" % "zoo-core-dist-all" % zooCoreVersion

val sparkMl = ("org.apache.spark" %% "spark-mllib" % sparkVersion)
  .cross(CrossVersion.for3Use2_13)

val sparkSql =
  ("org.apache.spark" %% "spark-sql" % sparkVersion).cross(CrossVersion.for3Use2_13)

val bigdlDlib =
  ("com.intel.analytics.bigdl" % bigdlDlibPackage % bigdlVersion from bigDlDlibArtifact)
    .excludeAll(ExclusionRule(organization = "org.scalactic", "scalactic"))
    .cross(CrossVersion.for3Use2_13)

val javaCppPresets =
  "org.bytedeco" % "onnx-platform" % javaCppOnnxPresetsVersion

val scopt = "com.github.scopt" %% "scopt" % scoptVersion

val jsoniterScalaCore = "com.github.plokhotnyuk.jsoniter-scala" %% "jsoniter-scala-core" % "2.23.3"

// Use the "provided" scope instead when the "compile-internal" scope is not supported
val jsoniterScalaMacros = "com.github.plokhotnyuk.jsoniter-scala" %% "jsoniter-scala-macros" % "2.23.3"

val sparkScala3 = "io.github.vincenzobaz" %% "spark-scala3" % "0.2.1"

val platform = "macosx"

val copyArtifacts = Seq(
  "*bigdl*",
  "*scopt*",
  "*javacpp*",
  "*jsoniter*",
  s"*onnx*",
  "*onnxruntime*",
  "*zoo-core*",
  "*scala3*"
)

lazy val copy = TaskKey[Unit]("copy")

import java.nio.file.{Files, Paths, StandardCopyOption}

copy := {
  val cp   = (Compile / fullClasspath).value
  val dist = Paths.get("dist")
  if (!Files.exists(dist))
    Files.createDirectory(dist)

  val providedDependencies = update
    .map { f =>
      copyArtifacts.map { name =>
        f.select(artifactFilter(name))
      }
    }
    .value
    .flatten

  providedDependencies.foreach { file =>
    Files.copy(file.toPath, new java.io.File(s"${dist}/${file.name}").toPath, StandardCopyOption.REPLACE_EXISTING)
  }
}

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
      sparkScala3,
      scopt
    )
  )

resolvers += "ossrh repository" at "https://oss.sonatype.org/content/repositories/snapshots/"
