// /v1/decide over loopback HTTP against a model that does not apply token constraints (Gemma4): a generated question is
// refused as a whole with a 400 question_type_unsupported before any question of the request is asked, the readout
// questions are served exactly as a model that applies constraints serves them, and the server answers requests after
// the refusals. No Engine: the backend stands in for GenerationService and answers what a program would.
#include "http_loopback_harness.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace infernix::serve;
using infernix::serve::test::LoopbackBackend;
using infernix::serve::test::LoopbackServer;

constexpr const char* kModel = "decide-test-model";

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// One token per byte (a digit is one token and "10" is two). `constraints` is what the backend reports of its model;
// run() answers a readout with two fixed log-probabilities and a generated question with the digit 4 at every digit step.
class DecideBackend final : public LoopbackBackend {
public:
    explicit DecideBackend(bool constraints) : LoopbackBackend(kModel), constraints_(constraints) {}

    bool supports_token_constraints() const override { return constraints_; }

    std::vector<infernix::TokenId> tokenize_text(std::string_view text) const override {
        std::vector<infernix::TokenId> ids;
        for (const unsigned char byte : text) { ids.push_back(static_cast<infernix::TokenId>(byte)); }
        return ids;
    }

    PreparedRequest prepare(const GenerationRequest& request, GenerationConsumerMode,
                            infernix::GenerationObservationOptions, std::function<bool()>) const override {
        ++prepared;
        constraint_ = request.token_constraint;
        PreparedRequest result;
        result.prompt_tokens = 8;
        return result;
    }

    GenerationOutcome run(PreparedRequest&, const StreamSink*, std::function<bool()>) override {
        ++ran;
        GenerationOutcome outcome;
        outcome.prompt_tokens = 8;
        if (constraint_.empty()) {
            outcome.readout = infernix::PromptReadout{.logprobs = {std::log(0.75F), std::log(0.25F)}};
            return outcome;
        }
        for (const std::vector<infernix::TokenId>& step : constraint_.steps) {
            infernix::ConstrainedDraw draw;
            draw.mass = 0.99F;
            draw.probabilities.assign(step.size(), 0.0F);
            std::size_t slot = 0;
            if (step.size() > 1) {
                slot = 4; // the digit 4
                draw.probabilities[slot] = 0.9F;
            } else {
                draw.probabilities[0] = 1.0F;
            }
            draw.token = step[slot];
            outcome.generated_token_ids.push_back(draw.token);
            outcome.constrained_draws.push_back(std::move(draw));
        }
        outcome.completion_tokens = static_cast<int>(outcome.generated_token_ids.size());
        return outcome;
    }

    mutable int prepared = 0;
    int ran              = 0;

private:
    bool constraints_;
    mutable infernix::TokenConstraint constraint_;
};

Json choice_question() {
    return Json{{"type", "choice"}, {"instructions", "Which queue?"},
                {"criteria", Json{{"payouts", "money out"}, {"billing", "money in"}}}};
}

Json generated_question(const char* type) {
    return Json{{"type", type}, {"instructions", "How many?"}, {"digits", 3}};
}

Json decide_body(Json questions) { return Json{{"model", kModel}, {"state", "A ticket."}, {"questions", std::move(questions)}}; }

std::string describe(const httplib::Result& result) {
    if (!result) { return " [no response: " + httplib::to_string(result.error()) + "]"; }
    return " [HTTP " + std::to_string(result->status) + " " + result->body.substr(0, 300) + "]";
}

Json body_of(const httplib::Result& result) {
    if (!result) { return Json(); }
    return Json::parse(result->body, nullptr, false);
}

bool refused_as_unsupported(const httplib::Result& result, const std::string& id, const std::string& type) {
    const Json body = body_of(result);
    if (!result || result->status != 400 || !body.is_object() || !body.contains("error")) { return false; }
    const Json& error = body["error"];
    const std::string message = error.value("message", std::string());
    return error.value("code", std::string()) == "question_type_unsupported" &&
           error.value("type", std::string()) == "invalid_request_error" &&
           error.value("param", std::string()) == "questions." + id && message.find(type) != std::string::npos &&
           message.find("noul, choice, score") != std::string::npos;
}

void test_generated_refused_and_server_keeps_serving() {
    DecideBackend backend(false);
    const LoopbackServer server(backend);

    const httplib::Result readout = server.post("/v1/decide", decide_body(Json{{"queue", choice_question()}}));
    check(readout && readout->status == 200, "a readout question is served on a model without constraints" + describe(readout));
    const Json served = body_of(readout);
    // The backend gives the first declared option (the JSON object's first key, "billing") 0.75.
    check(served.contains("answers") && served["answers"]["queue"].value("choice", std::string()) == "billing",
          "the readout answer names the likelier option" + describe(readout));
    check(backend.ran == 1, "the readout ran once");

    for (const char* type : {"number", "scalar"}) {
        const httplib::Result refused = server.post("/v1/decide", decide_body(Json{{"legs", generated_question(type)}}));
        check(refused_as_unsupported(refused, "legs", type),
              std::string("a ") + type + " is refused as question_type_unsupported with the decide error shape");
    }
    check(backend.prepared == 1 && backend.ran == 1, "a refused generated question asks nothing of the backend");

    const httplib::Result mixed =
        server.post("/v1/decide", decide_body(Json{{"queue", choice_question()}, {"legs", generated_question("number")}}));
    check(refused_as_unsupported(mixed, "legs", "number"), "a readout beside a generated question is refused as a whole");
    check(backend.prepared == 1 && backend.ran == 1, "the readout of a refused request is never asked");

    const httplib::Result malformed = server.post("/v1/decide", decide_body(Json{{"q", Json{{"type", "bogus"}, {"instructions", "x"}}}}));
    check(malformed && malformed->status == 400 &&
              body_of(malformed)["error"].value("code", std::string()) == "question_type_unsupported" &&
              body_of(malformed)["error"].value("param", std::string()) == "questions.q",
          "an unknown type keeps its own 400");

    const httplib::Result after = server.post("/v1/decide", decide_body(Json{{"queue", choice_question()}}));
    check(after && after->status == 200 && body_of(after) == served, "the readout after the refusals answers exactly as before");
    check(backend.ran == 2, "the readout after the refusals ran");
}

void test_capable_model_unchanged() {
    DecideBackend capable(true);
    DecideBackend without(false);
    const LoopbackServer capable_server(capable);
    const LoopbackServer without_server(without);

    const httplib::Result number = capable_server.post("/v1/decide", decide_body(Json{{"legs", generated_question("number")}}));
    check(number && number->status == 200 && body_of(number)["answers"]["legs"].value("number", -1) == 444,
          "a model that applies constraints still serves a number");

    const Json body = decide_body(Json{{"queue", choice_question()}});
    const httplib::Result a = capable_server.post("/v1/decide", body);
    const httplib::Result b = without_server.post("/v1/decide", body);
    check(a && b && a->status == 200 && b->status == 200 && body_of(a) == body_of(b),
          "a readout is answered identically with and without constraint support");
}

} // namespace

int main() {
    try {
        test_generated_refused_and_server_keeps_serving();
        test_capable_model_unchanged();
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "http decide refusal: all checks passed\n";
    return 0;
}
