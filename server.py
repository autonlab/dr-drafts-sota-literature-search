'''
    Flask server to run Dr. Draft's SOTA Literature Search in a web browser
'''
import re
from flask import Flask, render_template, request, jsonify
import main

app = Flask(__name__)
main.load_data()


def make_urls_active(text: str):
    ''' Make https URLs in the text active '''
    url_pattern = re.compile(r'(https://[^\s<]+)')

    def replace_with_link(pattern_match: str):
        url = pattern_match.group(1)
        style = 'style="color: lightblue; text-decoration: underline;"'
        return f'<a href="{url}" target="_blank" {style}>{url}</a>'

    return url_pattern.sub(replace_with_link, text)


# Route for the UI
@app.route('/')
def index():
    ''' Render the index page '''
    return render_template('index.html')


# API to execute the script
@app.route('/run-script', methods=['POST'])
def run_script():
    ''' Run the script and return the output '''
    data = request.json
    prompt = data.get('arg1', 'missing prompt')
    k = data.get('arg2', '1')
    k = int(k)

    result = main.run_dr_drafts(prompt,
                                k,
                                'flask_output.csv',
                                'Flask user prompt')
    html_output = make_urls_active(result)
    return jsonify({"message": html_output})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
