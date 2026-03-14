const { ESLint } = require("eslint");

(async function main() {
    const eslint = new ESLint();
    const results = await eslint.lintFiles(["src/**/*.tsx"]);

    // Format the results.
    const formatter = await eslint.loadFormatter("stylish");
    const resultText = await formatter.format(results);

    // Output it.
    console.log(resultText);
})().catch((error) => {
    process.exitCode = 1;
    console.error(error);
});
