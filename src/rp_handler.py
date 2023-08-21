'''
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
'''

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict
import base64


MODEL = predict.Predictor()
MODEL.setup()


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    job_input = job['input']

    def download_uri(audio):
        if audio[:4] == "bin:":
            # Raw binary data, in a string
            data = bytes(audio[4:], "iso-8859-1")
            return "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii")
        elif audio[:5] != "data:":
            return download_files_from_urls(job['id'], [audio])[0]
        else:
            return audio

    def predict_uri(audio):
        return MODEL.predict(
            audio=audio,
            model_name=job_input["model"],
            transcription=job_input["transcription"],
            translate=job_input["translate"],
            language=job_input["language"],
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            word_timestamps=job_input["word_timestamps"],
            vad_filter=job_input["vad_filter"],
            detailed=job_input["detailed"],
        )

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    with rp_debugger.LineTimer('download_step'):
        if job_input['audio'] == '':
            job_input['audios'] = list(map(download_uri, job_input['audios']))
        else:
            job_input['audio'] = download_uri(job_input['audio'])

    with rp_debugger.LineTimer('prediction_step'):
        if job_input['audio'] == '':
            whisper_results = list(map(predict_uri, job_input['audios']))
        else:
            whisper_results = predict_uri(job_input['audio'])

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})
