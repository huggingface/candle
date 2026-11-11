# Examples

**The examples shown here will not necessarily work for the latest stable release of Vulkano. By default, GitHub shows the most recent unstable revision, rather than the last stable release. You can switch GitHub to show the examples for stable releases as follows.**

## Selecting the version tag:

|||
| ----------- | ----------- |
| 1. Find the branch/tag selection box and click it. | ![change_tag_1](https://user-images.githubusercontent.com/5326321/155281242-c4477b00-c036-47f1-bb30-d9dabf91c56e.png) |
| 2. Select click on the tags tab. | ![change_tag_2](https://user-images.githubusercontent.com/5326321/155281245-f95ba940-6514-47a9-85f0-2174b0e78c07.png) |
| 3. Click on the desired version. | ![change_tag_3](https://user-images.githubusercontent.com/5326321/155281246-96abd6d4-e61b-47c8-b4c4-6f5b1610ee23.png) |
| 4. After selecting it should look like this. | ![change_tag_4](https://user-images.githubusercontent.com/5326321/155281247-5cd1ed27-d825-44d8-a390-abb5cbddcd7b.png) |

## Running the examples:

```sh
cargo run --bin <example>
```

## Example:

```sh
cargo run --bin triangle
```

If you want to compare performances with other libraries, you should pass the `--release` flag as
well. Rust is pretty slow in debug mode.
