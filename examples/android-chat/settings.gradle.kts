// Sample chat app over the packaged SAPIENT Android module.
//
// Requires the generated module to exist first:
//   ./scripts/package-android.sh        (from the repo root)
// then:
//   ./gradlew :app:assembleDebug        (from this directory)
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "sapient-chat"
include(":app")
// The packaged engine module (jniLibs + generated Kotlin + JNA dep).
include(":sapient-android")
project(":sapient-android").projectDir = file("../../dist/mobile/sapient-android")
