import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  Button,
  Image,
  ActivityIndicator,
  StyleSheet,
  TextInput,
  Alert,
  Platform,
  ScrollView,
} from "react-native";
import * as ImagePicker from "expo-image-picker";

export default function Index() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<null | { predicted?: string; recyclable?: boolean; details?: any }>(
    null
  );
  const [backendUrl, setBackendUrl] = useState<string>(
    // Default value; replace with your machine IP and port when running on a physical device
    Platform.OS === "android" ? "http://10.0.2.2:8000/predict" : "http://localhost:8000/predict"
  );

  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== "granted") {
        Alert.alert("Permission required", "Permission to access media library is required.");
      }
    })();
  }, []);

  async function pickImage() {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        setImageUri(result.assets[0].uri);
        setResult(null);
      }
    } catch (e) {
      Alert.alert("Error", "Could not open image picker.");
    }
  }

  async function uploadImage() {
    if (!imageUri) {
      Alert.alert("No image", "Please pick an image first.");
      return;
    }

    setUploading(true);
    setResult(null);

    try {
      const uriParts = imageUri.split(".");
      const fileType = uriParts[uriParts.length - 1];

      const formData = new FormData();
      // @ts-ignore - React Native FormData file shape
      formData.append("file", {
        uri: imageUri,
        name: `photo.${fileType}`,
        type: `image/${fileType === "jpg" ? "jpeg" : fileType}`,
      });

      const resp = await fetch(backendUrl, {
        method: "POST",
        body: formData,
        headers: {
          // Do not set Content-Type; let fetch set the multipart boundary
          Accept: "application/json",
        },
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server error ${resp.status}: ${text}`);
      }

      const json = await resp.json();
      // Expecting something like: { predicted: 'plastic_bottle', recyclable: true }
      setResult({ predicted: json.predicted || json.prediction || json.label, recyclable: json.recyclable ?? json.recyclability ?? json.is_recyclable, details: json });
    } catch (e: any) {
      Alert.alert("Upload failed", e.message || String(e));
    } finally {
      setUploading(false);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Waste Material Classifier</Text>

      <Text style={styles.label}>Backend predict URL</Text>
      <TextInput
        style={styles.input}
        value={backendUrl}
        onChangeText={setBackendUrl}
        placeholder="http://192.168.x.x:8000/predict"
        autoCapitalize="none"
        keyboardType="url"
      />

      <View style={styles.buttonRow}>
        <View style={styles.buttonWrap}>
          <Button title="Pick Image" onPress={pickImage} />
        </View>
        <View style={styles.buttonWrap}>
          <Button title="Upload" onPress={uploadImage} disabled={!imageUri || uploading} />
        </View>
      </View>

      {imageUri ? (
        <Image source={{ uri: imageUri }} style={styles.image} />
      ) : (
        <View style={styles.placeholder}>
          <Text style={styles.placeholderText}>No image selected</Text>
        </View>
      )}

      {uploading && <ActivityIndicator size="large" color="#007aff" style={{ marginTop: 12 }} />}

      {result && (
        <View style={styles.resultBox}>
          <Text style={styles.resultText}>Prediction: {result.predicted ?? "—"}</Text>
          <Text style={styles.resultText}>
            Recyclable: {result.recyclable === undefined ? "Unknown" : result.recyclable ? "Yes" : "No"}
          </Text>
          <Text style={styles.detailsLabel}>Raw response:</Text>
          <Text style={styles.detailsText}>{JSON.stringify(result.details)}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    alignItems: "stretch",
  },
  title: {
    fontSize: 22,
    fontWeight: "600",
    marginBottom: 12,
    textAlign: "center",
  },
  label: {
    fontSize: 14,
    marginBottom: 6,
  },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 6,
    padding: 8,
    marginBottom: 12,
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  buttonWrap: {
    flex: 1,
    marginHorizontal: 6,
  },
  image: {
    width: "100%",
    height: 300,
    borderRadius: 8,
    marginTop: 8,
  },
  placeholder: {
    height: 300,
    borderRadius: 8,
    backgroundColor: "#f2f2f2",
    justifyContent: "center",
    alignItems: "center",
  },
  placeholderText: {
    color: "#666",
  },
  resultBox: {
    marginTop: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 8,
    backgroundColor: "#fff",
  },
  resultText: {
    fontSize: 16,
    marginBottom: 6,
  },
  detailsLabel: { fontSize: 12, color: "#444", marginTop: 8 },
  detailsText: { fontSize: 11, color: "#333", marginTop: 4 },
});

