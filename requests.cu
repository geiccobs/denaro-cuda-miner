#include <string.h>
#include <curl/curl.h>
#include <json-c/json.h>

#include "requests.cuh"

#define CHUNK_SIZE 2048

typedef struct {
    unsigned char *buffer;
    size_t len;
    size_t buffer_length;
} GetRequest;

size_t write_callback(char *ptr, size_t size, size_t nmemb, void *userdata) {
    GetRequest *req = (GetRequest *) userdata;
    size_t new_len = req->len + size * nmemb;

    if (new_len > req->buffer_length) {
        req->buffer_length = new_len + CHUNK_SIZE;
        req->buffer = (unsigned char *) realloc(req->buffer, req->buffer_length);
    }

    memcpy(req->buffer + req->len, ptr, size * nmemb);
    req->len = new_len;
    req->buffer[req->len] = '\0';

    return size * nmemb;
}

json_object *curl_get(const char *url) {
    CURL *curl;
    CURLcode res;

    GetRequest req = {NULL, 0, 0};
    json_object *response = NULL;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "GET");
        curl_easy_setopt(curl, CURLOPT_URL, url);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &req);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            exit(1);
        }

        response = json_tokener_parse((const char *) req.buffer);
        curl_easy_cleanup(curl);
    }

    return response;
}

json_object *curl_post(const char *url, const char *data) {
    CURL *curl;
    CURLcode res;

    GetRequest req = {NULL, 0, 0};
    json_object *response = NULL;

    curl = curl_easy_init();
    if (curl) {
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
        curl_easy_setopt(curl, CURLOPT_URL, url);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &req);

        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            exit(1);
        }

        response = json_tokener_parse((const char *) req.buffer);
        curl_easy_cleanup(curl);
    }

    return response;
}

Share share(const char *poolUrl, const char *hash, const char pending_transactions_hashes[512][64 + 1], size_t pending_transactions_count, uint id) {
    char url[512];
    strcpy(url, poolUrl);
    strcat(url, "share");

    json_object *jobj = json_object_new_object();
    json_object_object_add(jobj, "block_content", json_object_new_string(hash));

    json_object *jarray = json_object_new_array();
    for (int i = 0; i < pending_transactions_count; ++i) {
        json_object_array_add(jarray, json_object_new_string(pending_transactions_hashes[i]));
    }

    json_object_object_add(jobj, "txs", jarray);
    json_object_object_add(jobj, "id", json_object_new_int(id));

    //printf("request: %s\n", json_object_to_json_string_ext(jobj, JSON_C_TO_STRING_PRETTY));
    json_object *response = curl_post(url, json_object_to_json_string_ext(jobj, JSON_C_TO_STRING_PRETTY));
    //printf("response: %s\n", json_object_to_json_string_ext(response, JSON_C_TO_STRING_PRETTY));

    Share share;
    share.ok = json_object_get_boolean(json_object_object_get(response, "ok"));
    if (!share.ok) {
        strcpy(share.error, json_object_get_string(json_object_object_get(response, "error")));
    }

    free(jobj);
    free(response);
    free(jarray);

    return share;
}

MiningInfo get_mining_info(const char *nodeUrl) {
    MiningInfo response;

    char url[512];
    strcpy(url, nodeUrl);
    strcat(url, "get_mining_info");

    json_object *mining_info = curl_get(url);
    if (mining_info == NULL) {
        response.ok = false;
        return response;
    }

    json_object *result = json_object_object_get(mining_info, "result");
    json_object *last_block = json_object_object_get(result, "last_block");

    response.ok = json_object_get_boolean(json_object_object_get(mining_info, "ok"));

    response.result.difficulty = json_object_get_double(json_object_object_get(result, "difficulty"));
    strcpy(response.result.merkle_root, json_object_get_string(json_object_object_get(result, "merkle_root")));

    response.result.pending_transactions_count = json_object_array_length(json_object_object_get(result, "pending_transactions_hashes"));
    if (response.result.pending_transactions_count > 512) {
        response.result.pending_transactions_count = 512;
    }

    for (int i = 0; i < response.result.pending_transactions_count; ++i) {
        strcpy(response.result.pending_transactions_hashes[i], json_object_get_string(
                json_object_array_get_idx(json_object_object_get(result, "pending_transactions_hashes"), i)));
    }

    response.result.last_block.id = json_object_get_int(json_object_object_get(last_block, "id"));
    strcpy(response.result.last_block.hash, json_object_get_string(json_object_object_get(last_block, "hash")));
    strcpy(response.result.last_block.address, json_object_get_string(json_object_object_get(last_block, "address")));
    response.result.last_block.random = json_object_get_int(json_object_object_get(last_block, "random"));
    response.result.last_block.difficulty = json_object_get_double(json_object_object_get(last_block, "difficulty"));
    response.result.last_block.reward = json_object_get_double(json_object_object_get(last_block, "reward"));
    response.result.last_block.timestamp = json_object_get_int(json_object_object_get(last_block, "timestamp"));
    strcpy(response.result.last_block.content, json_object_get_string(json_object_object_get(last_block, "content")));

    return response;
}

char *get_mining_address(const char *poolUrl, const char *address) {
    char url[512];
    strcpy(url, poolUrl);
    strcat(url, "get_mining_address?address=");
    strcat(url, address);

    return (char *) json_object_get_string(json_object_object_get(curl_get(url), "address"));
}