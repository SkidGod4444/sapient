plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "so.openhorizon.sapient.chat"
    compileSdk = 35

    defaultConfig {
        applicationId = "so.openhorizon.sapient.chat"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        // The packaged module ships arm64-v8a (+ x86_64 with --emulator).
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
}

dependencies {
    // The packaged engine (see settings.gradle.kts / scripts/package-android.sh).
    implementation(project(":sapient-android"))

    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
}
