package ru.alenavir;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.KubernetesClientBuilder;
import okhttp3.*;

import java.io.IOException;

public class ReconcileLoop {

    private static final String PREDICTOR_URL = "http://localhost:8000/predict/auto";
    private static final String NAMESPACE = "default";
    private static final String DEPLOYMENT_NAME = "demo-app";
    private static final int MIN_REPLICAS = 2;
    private static final int MAX_REPLICAS = 20;
    private static final long RECONCILE_INTERVAL_MS = 30_000;

    private static final OkHttpClient http = new OkHttpClient();
    private static final ObjectMapper mapper = new ObjectMapper();

    public static void main(String[] args) {
        try (KubernetesClient k8s = new KubernetesClientBuilder().build()) {

            System.out.println("Reconcile loop запущен. Интервал: " + RECONCILE_INTERVAL_MS + "ms");

            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("Shutting down reconcile loop...");
            }));

            while (!Thread.currentThread().isInterrupted()) {
                try {
                    reconcile(k8s);
                } catch (Exception e) {
                    System.err.println("Ошибка в цикле reconcile: " + e.getMessage());
                    e.printStackTrace();
                }

                try {
                    Thread.sleep(RECONCILE_INTERVAL_MS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    System.out.println("Reconcile loop interrupted, exiting...");
                    break;
                }
            }
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void reconcile(KubernetesClient k8s) throws IOException {
        int recommendedReplicas = fetchRecommendedReplicas();
        int clamped = Math.max(MIN_REPLICAS, Math.min(MAX_REPLICAS, recommendedReplicas));

        Integer currentReplicasObj = k8s.apps().deployments()
                .inNamespace(NAMESPACE)
                .withName(DEPLOYMENT_NAME)
                .get()
                .getSpec()
                .getReplicas();

        int currentReplicas = currentReplicasObj != null ? currentReplicasObj : 0;

        System.out.printf("Прогноз рекомендует: %d | Текущее: %d%n", clamped, currentReplicas);

        if (clamped != currentReplicas) {
            k8s.apps().deployments()
                    .inNamespace(NAMESPACE)
                    .withName(DEPLOYMENT_NAME)
                    .scale(clamped);
            System.out.println(">>> Заскейлили с " + currentReplicas + " до " + clamped);
        } else {
            System.out.println(">>> Изменений не требуется");
        }
    }

    private static int fetchRecommendedReplicas() throws IOException {
        MediaType mediaType = MediaType.get("application/json");

        RequestBody body = RequestBody.create(
                "{\"horizon_minutes\": 5}",
                mediaType
        );
        Request request = new Request.Builder()
                .url(PREDICTOR_URL)
                .post(body)
                .build();

        Response response = http.newCall(request).execute();
        ResponseBody responseBody = response.body();
        try {
            if (responseBody == null) {
                throw new IOException("Response body is null");
            }
            String json = responseBody.string();
            JsonNode node = mapper.readTree(json);
            return node.get("recommended_replicas").asInt();
        } finally {
            if (responseBody != null) {
                responseBody.close();
            }
        }
    }
}