Rust implementation of Reed-Solomon encoding and decoding (Euclidean algorithm for decoding).

Try:

```console
$ echo "hello, world!" | cargo run -- --encode 12 | sed 's/hello, world/bye for now!/' | cargo run -- --decode 12
```