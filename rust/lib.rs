use abi_stable::std_types::RVec;
use abi_stable::StableAbi;

#[derive(Clone, Debug, PartialEq, StableAbi)]
#[repr(transparent)]
pub struct BasicBlock<T>(pub RVec<T>);

impl<T> BasicBlock<T> {
    pub fn new() -> Self {
        Self(RVec::new())
    }
}
