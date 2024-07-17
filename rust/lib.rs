// 中をpubにしているのは仮。
// 将来的にはread streamやwrite streamのみを外部インターフェースとして提供する
#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlock<T>(pub Vec<T>);

impl<T> BasicBlock<T> {
    pub fn new() -> Self {
        Self(Vec::new())
    }
}
