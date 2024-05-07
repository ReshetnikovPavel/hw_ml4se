import torch.nn.functional as F
from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
)
import torch
from datasets import load_dataset, concatenate_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizerFast.from_pretrained(
    "microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
model.to(device)


def compute_embedding(texts: list[str]) -> torch.tensor:
    output = [tokenizer(
        text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512) for text in texts]
    tokens = torch.tensor([x['input_ids'] for x in output]).to(device)
    attention_mask = torch.tensor([x['attention_mask']
                                  for x in output]).to(device)
    with torch.no_grad():
        return model(tokens, attention_mask=attention_mask).last_hidden_state


def are_clones(first: list[str], second: list[str]):
    emb1 = compute_embedding(first)
    emb2 = compute_embedding(second)
    cos_sim = F.cosine_similarity(emb1[:, 0], emb2[:, 0])
    return cos_sim


text1 = ["""
@Override public String getMessageDigest() throws SarasvatiLoadException { if (messageDigest == null) { Collections.sort(nodes); Collections.sort(externals); try { MessageDigest digest = MessageDigest.getInstance("SHA1"); digest.update(name.getBytes()); for (XmlNode node : nodes) { node.addToDigest(digest); } for (XmlExternal external : externals) { external.addToDigest(digest); } messageDigest = SvUtil.getHexString(digest.digest()); } catch (NoSuchAlgorithmException nsae) { throw new SarasvatiException("Unable to load SHA1 algorithm", nsae); } } return messageDigest; } 
""",
         ]

text2 = ["""
 static HttpURLConnection connect(String url, String method, String contentType, String content, int timeoutMillis) throws ProtocolException, IOException, MalformedURLException, UnsupportedEncodingException { HttpURLConnection conn = (HttpURLConnection) (new URL(url).openConnection()); conn.setRequestMethod(method); conn.setConnectTimeout(timeoutMillis); byte[] bContent = null; if (content != null && content.length() > 0) { conn.setDoOutput(true); conn.setRequestProperty("Content-Type", contentType); bContent = content.getBytes("UTF-8"); conn.setFixedLengthStreamingMode(bContent.length); } conn.connect(); if (bContent != null) { OutputStream os = conn.getOutputStream(); os.write(bContent); os.flush(); os.close(); } return conn; } 
""",

         ]

# print(are_clones(text1, text2))

# dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")


def calculate_metrics(actual, expected):
    TP = sum((a == 1 and e == 1) for a, e in zip(actual, expected))
    TN = sum((a == 0 and e == 0) for a, e in zip(actual, expected))
    FP = sum((a == 0 and e == 1) for a, e in zip(actual, expected))
    FN = sum((a == 1 and e == 0) for a, e in zip(actual, expected))

    total = len(actual)
    right_answers = TP + TN
    percentage_right = (right_answers / total) * 100

    return TP, TN, FP, FN, percentage_right


def trim_text(text):
    first_brace_index = text.find('{')
    last_brace_index = text.rfind('}')

    if first_brace_index != -1 and last_brace_index != -1:
        trimmed_text = text[first_brace_index:last_brace_index+1]
        return trimmed_text
    else:
        return text


# test = dataset['test']


def make_50_50(dataset):
    true_dataset = dataset.filter(lambda example: example['label'] == True)
    false_dataset = dataset.filter(lambda example: example['label'] == False)
    true_examples = true_dataset.select(range(50))
    false_examples = false_dataset.select(range(50))
    balanced_examples = concatenate_datasets([true_examples, false_examples])
    balanced_examples = balanced_examples.shuffle(seed=42)
    return balanced_examples


# test = make_50_50(test)
# print(test)
#
# batch_size = 10
# end = 100
# res = []
#
# for i in range(0, end, batch_size):
#     print(i, '/', end)
#     func1 = test['func1'][i:i+batch_size]
#     func2 = test['func2'][i:i+batch_size]
#     sim = list(are_clones(func1, func2))
#     res.extend(sim)
#     print(sim, test['label'][i:i+batch_size])
#
# expected = test['label'][:100]
#
# best_score = 0
# threshold = 0
# actual = []
# print(expected)
# for i in range(0, 1000):
#     t = i / 10
#     # print(t)
#     actual = [1 if x > t else 0 for x in res]
#     # tp, tn, fp, fn, percentage = calculate_metrics(actual, expected)
#     if len(set(actual)) == 1:
#         continue
#     score = roc_auc_score(actual, expected)
#     if score > best_score:
#         best_score = score
#         threshold = t
#
# print("TRESHOLD::: ", threshold)
# actual = [1 if x > 0.89 else 0 for x in res]
# for i, e in enumerate(actual):
#     print(e, expected[i])
# print(calculate_metrics(actual, expected))

type2_clones = [
    {
        "func1":  """
public double calculateArea(double width, double height) {
    return width * height;
}
            """,
        "func2":  """
public double computeArea(double length, double breadth) {
    return length * breadth;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public int findMax(int a, int b) {
    return (a > b) ? a : b;
}
            """,
        "func2":  """
public int getMax(int x, int y) {
    return (x > y) ? x : y;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public boolean isPrime(int num) {
    for (int i = 2; i < num; i++) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}
            """,
        "func2":  """
public boolean checkPrime(int number) {
    for (int j = 2; j < number; j++) {
        if (number % j == 0) {
            return false;
        }
    }
    return true;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public String reverseString(String str) {
    String reversed = "";
    for (int i = str.length() - 1; i >= 0; i--) {
        reversed += str.charAt(i);
    }
    return reversed;
}
            """,
        "func2":  """
public String flipString(String input) {
    String result = "";
    for (int j = input.length() - 1; j >= 0; j--) {
        result += input.charAt(j);
    }
    return result;
}
            """,
        "label": True,
    },

    {
        "func1":  """
public int sumArray(int[] arr) {
    int sum = 0;
    for (int i : arr) {
        sum += i;
    }
    return sum;
}
            """,
        "func2":  """
public int totalArray(int[] array) {
    int total = 0;
    for (int j : array) {
        total += j;
    }
    return total;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public double findAverage(int[] numbers) {
    int sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    return (double) sum / numbers.length;
}
            """,
        "func2":  """
public double calculateAverage(int[] data) {
    int total = 0;
    for (int value : data) {
        total += value;
    }
    return (double) total / data.length;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public boolean isPalindrome(String word) {
    String reversed = "";
    for (int i = word.length() - 1; i >= 0; i--) {
        reversed += word.charAt(i);
    }
    return word.equals(reversed);
}
            """,
        "func2":  """
public boolean checkPalindrome(String text) {
    String flipped = "";
    for (int j = text.length() - 1; j >= 0; j--) {
        flipped += text.charAt(j);
    }
    return text.equals(flipped);
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public int findFactorial(int n) {
    int factorial = 1;
    for (int i = 1; i <= n; i++) {
        factorial *= i;
    }
    return factorial;
}
            """,
        "func2":  """
public int calculateFactorial(int number) {
    int result = 1;
    for (int j = 1; j <= number; j++) {
        result *= j;
    }
    return result;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public boolean isEven(int num) {
    return num % 2 == 0;
}
            """,
        "func2":  """
public boolean checkEven(int number) {
    return number % 2 == 0;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public double convertCelsiusToFahrenheit(double celsius) {
    return (celsius * 9/5) + 32;
}
            """,
        "func2":  """
public double celsiusToFahrenheit(double tempCelsius) {
    return (tempCelsius * 9/5) + 32;
}
            """,
        "label":  True,
    },
]

type3_clones = [
    {
        "func1": """
public double calculateArea(double width, double height) {
    return width * height;
}
            """,
        "func2": """
public double computeArea(double length, double breadth) {
    if (length <= 0 || breadth <= 0) {
        throw new IllegalArgumentException("Dimensions must be positive.");
    }
    return length * breadth;
}
            """,
        "label": True,
    },
    {
        "func1": """
public double calculateArea(double width, double height) {
    return width * height;
}
            """,
        "func2": """
public double computeArea(double length, double breadth) {
    if (length <= 0 || breadth <= 0) {
        throw new IllegalArgumentException("Dimensions must be positive.");
    }
    return length * breadth;
}
            """,
        "label": True,
    },

    {
        "func1": """
public int findMax(int a, int b) {
    return (a > b) ? a : b;
}
            """,
        "func2": """
public int getMax(int x, int y) {
    int max = (x > y) ? x : y;
    System.out.println("Max value is: " + max);
    return max;
}
            """,
        "label": True,
    },

    {
        "func1": """
public boolean isPrime(int num) {
    for (int i = 2; i < num; i++) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}
            """,
        "func2": """
public boolean checkPrime(int number) {
    if (number <= 1) {
        return false;
    }
    for (int i = 2; i * i <= number; i++) {
        if (number % i == 0) {
            return false;
        }
    }
    return true;
}
            """,
        "label": True,
    },

    {
        "func1": """
public String reverseString(String str) {
    String reversed = "";
    for (int i = str.length() - 1; i >= 0; i--) {
        reversed += str.charAt(i);
    }
    return reversed;
}
            """,
        "func2": """
public String flipString(String input) {
    if (input == null) {
        return null;
    }
    String result = "";
    for (int j = input.length() - 1; j >= 0; j--) {
        result += input.charAt(j);
    }
    return result;
}
            """,
        "label": True,
    },

    {
        "func1": """
public int sumArray(int[] arr) {
    int sum = 0;
    for (int i : arr) {
        sum += i;
    }
    return sum;
}
            """,
        "func2": """
public int totalArray(int[] array) {
    int total = 0;
    for (int j : array) {
        if (j < 0) {
            throw new IllegalArgumentException("Negative numbers are not allowed.");
        }
        total += j;
    }
    return total;
}
            """,
        "label": True,
    },

    {
        "func1": """
public double findAverage(int[] numbers) {
    int sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    return (double) sum / numbers.length;
}
            """,
        "func2": """
public double calculateAverage(int[] data) {
    if (data.length == 0) {
        throw new IllegalArgumentException("Array must not be empty.");
    }
    int total = 0;
    for (int value : data) {
        total += value;
    }
    return (double) total / data.length;
}
            """,
        "label": True,
    },

    {
        "func1": """
public boolean isPalindrome(String word) {
    String reversed = "";
    for (int i = word.length() - 1; i >= 0; i--) {
        reversed += word.charAt(i);
    }
    return word.equals(reversed);
}
            """,
        "func2": """
public boolean checkPalindrome(String text) {
    String flipped = "";
    for (int j = text.length() - 1; j >= 0; j--) {
        flipped += text.charAt(j);
    }
    return text.equalsIgnoreCase(flipped);
}
            """,
        "label": True,
    },

    {
        "func1": """
public int findFactorial(int n) {
    int factorial = 1;
    for (int i = 1; i <= n; i++) {
        factorial *= i;
    }
    return factorial;
}
            """,
        "func2": """
public int calculateFactorial(int number) {
    if (number < 0) {
        throw new IllegalArgumentException("Factorial is not defined for negative numbers.");
    }
    int result = 1;
    for (int j = 1; j <= number; j++) {
        result *= j;
    }
    return result;
}
            """,
        "label": True,
    },

    {
        "func1": """
public boolean isEven(int num) {
    return num % 2 == 0;
}
            """,
        "func2": """
public boolean checkEven(int number) {
    if (number % 2 == 0) {
        return true;
    } else {
        return false;
    }
}
            """,
        "label": True,
    },

    {
        "func1": """
public double convertCelsiusToFahrenheit(double celsius) {
    return (celsius * 9/5) + 32;
}
            """,
        "func2": """
public double celsiusToFahrenheit(double tempCelsius) {
    if (tempCelsius < -273.15) {
        throw new IllegalArgumentException("Temperature below absolute zero is not possible.");
    }
    return (tempCelsius * 9/5) + 32;
}
            """,
        "label": True,
    },
]

type4_clones = [

        {
            "func1": """
public int factorialIterative(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}
            """,
            "func2": """
public int factorialRecursive(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorialRecursive(n - 1);
    }
}
            """,
            "label": True,
        },

        {
            "func1": """
public int findMaxIterative(int[] array) {
    int max = array[0];
    for (int i : array) {
        if (i > max) {
            max = i;
        }
    }
    return max;
}
            """,
            "func2": """
import java.util.Arrays;

public int findMaxStream(int[] array) {
    return Arrays.stream(array).max().orElse(Integer.MIN_VALUE);
}
            """,
            "label": True,
        },
        {
            "func1": """
public boolean isPalindromeIterative(String str) {
    int start = 0;
    int end = str.length() - 1;
    while (start < end) {
        if (str.charAt(start) != str.charAt(end)) {
            return false;
        }
        start++;
        end--;
    }
    return true;
}
            """,
            "func2": """
public boolean isPalindromeStringBuilder(String str) {
    return str.equals(new StringBuilder(str).reverse().toString());
}
            """,
            "label": True,
        },

        {
            "func1": """
public int sumArrayIterative(int[] array) {
    int sum = 0;
    for (int i : array) {
        sum += i;
    }
    return sum;
}
            """,
            "func2": """
import java.util.Arrays;

public int sumArrayStream(int[] array) {
    return Arrays.stream(array).sum();
}
            """,
            "label": True,
        },

        {
            "func1": """
public String reverseStringIterative(String str) {
    String reversed = "";
    for (int i = str.length() - 1; i >= 0; i--) {
        reversed += str.charAt(i);
    }
    return reversed;
}
            """,
            "func2": """
public String reverseStringStringBuilder(String str) {
    return new StringBuilder(str).reverse().toString();
}
            """,
            "label": True,
        },

        {
            "func1": """
public boolean isPrimeIterative(int num) {
    if (num <= 1) {
        return false;
    }
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}
            """,
            "func2": """
import java.util.stream.IntStream;

public boolean isPrimeStream(int num) {
    return num > 1 && IntStream.rangeClosed(2, (int) Math.sqrt(num)).noneMatch(i -> num % i == 0);
}
            """,
            "label": True,
        },
        {
            "func1": """
public double findAverageIterative(int[] array) {
    int sum = 0;
    for (int i : array) {
        sum += i;
    }
    return (double) sum / array.length;
}
            """,
            "func2": """

import java.util.Arrays;

public double findAverageStream(int[] array) {
    return Arrays.stream(array).average().orElse(0.0);
}
            """,
            "label": True,
        },

        {
            "func1": """
public boolean containsValueIterative(int[] array, int value) {
    for (int i : array) {
        if (i == value) {
            return true;
        }
    }
    return false;
}
            """,
            "func2": """
import java.util.Arrays;

public boolean containsValueStream(int[] array, int value) {
    return Arrays.stream(array).anyMatch(i -> i == value);
}
            """,
            "label": True,
        },
        {
            "func1": """
public void bubbleSort(int[] array) {
    int n = array.length;
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (array[j] > array[j+1]) {
                // swap array[j+1] and array[j]
                int temp = array[j];
                array[j] = array[j+1];
                array[j+1] = temp;
            }
        }
    }
}
            """,
            "func2": """
import java.util.Arrays;
import java.util.Comparator;

public void sortArrayStream(int[] array) {
    int[] sortedArray = Arrays.stream(array)
                                 .boxed()
                                 .sorted(Comparator.naturalOrder())
                                 .mapToInt(i -> i)
                                 .toArray();
    System.arraycopy(sortedArray, 0, array, 0, array.length);
}
            """,
            "label": True,
        },
        {
            "func1": """
import java.util.ArrayList;
import java.util.List;

public List<String> toUpperCaseIterative(List<String> list) {
    List<String> upperCaseList = new ArrayList<>();
    for (String str : list) {
        upperCaseList.add(str.toUpperCase());
    }
    return upperCaseList;
}
            """,
            "func2": """
import java.util.List;
import java.util.stream.Collectors;

public List<String> toUpperCaseStream(List<String> list) {
    return list.stream()
               .map(String::toUpperCase)
               .collect(Collectors.toList());
}
            """,
            "label": True,
        },
]

func1 = [x["func1"] for x in type2_clones]
func2 = [x["func2"] for x in type4_clones]
labels = [x["label"] for x in type4_clones]
print(are_clones(func1, func2))
