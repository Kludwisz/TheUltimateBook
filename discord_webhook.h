#ifndef DISCORD_WEBHOOK_H
#define DISCORD_WEBHOOK_H

#include <curl/curl.h>
#include <string>

inline void send_result(int num_enchants, uint64_t lo, uint64_t hi, std::string credit, int gpu_id) {
    const std::string webhook =
        "https://discord.com/api/webhooks/1517214908410892398/YLdVmseUJn2ItFpfeb9r4VeRy_gd3tlTWeMSB6vrZnWt9wkdf7Iq9AXIVt0dwAwNU7Wx";

    const int64_t slo = static_cast<int64_t>(lo);
    const int64_t shi = static_cast<int64_t>(hi);

    const std::string json = 
        R"({"content":"lo = )" + std::to_string(slo) + ", hi = " 
        + std::to_string(shi) + ", enchants = " + std::to_string(num_enchants) + " found by: " + credit + " (gpu #" + std::to_string(gpu_id) + ") " + R"("})";

    curl_global_init(CURL_GLOBAL_DEFAULT);

    CURL* curl = curl_easy_init();
    if (curl) {
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, webhook.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json.c_str());

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            fprintf(stderr, "curl error: %s\n",
                    curl_easy_strerror(res));
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();
}

#endif