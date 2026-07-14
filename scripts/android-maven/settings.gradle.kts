// Minimal Maven-publishing scaffold for the GENERATED Android module at
// dist/mobile/sapient-android (run scripts/package-android.sh first — or
// unzip a release's sapient-android.zip there). Publishes the AAR + POM
// into a directory laid out as a Maven repository — normally a clone of
// the git-hosted repo https://github.com/openhorizon-labs/sapient-android.
//
// No Gradle wrapper of its own — invoke via the example app's wrapper:
//
//   examples/android-chat/gradlew -p scripts/android-maven \
//     :sapient-android:publishReleasePublicationToDistRepository \
//     -PsapientVersion=0.6.0 \
//     -PsapientMavenDir=/abs/path/to/sapient-android-clone
//
// Consumers then need only:
//   repositories { maven { url = uri("https://raw.githubusercontent.com/openhorizon-labs/sapient-android/main") } }
//   dependencies { implementation("so.openhorizon:sapient:<version>") }
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

rootProject.name = "sapient-android-maven"
include(":sapient-android")
project(":sapient-android").projectDir = file("../../dist/mobile/sapient-android")
