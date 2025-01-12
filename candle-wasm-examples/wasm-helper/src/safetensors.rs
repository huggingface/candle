use std::borrow::Cow;
use std::collections::HashMap;

use safetensors::{tensor::TensorInfo, Dtype, SafeTensorError, View};
use serde::{ser::SerializeMap, Deserialize, Serialize};
use serde::{Deserializer, Serializer};

use crate::{generic_error::GenericResult, opfs::Blob};

const MAX_HEADER_SIZE: usize = 100_000_000;

/// A structure owning some metadata to lookup tensors on a shared `data`
/// byte-buffer (not owned).
#[derive(Debug)]
pub struct SafeTensors {
    metadata: Metadata,
    data: Blob,
    data_offset: usize,
}

impl SafeTensors {
    /// Given a byte-buffer representing a chunk of the byte array
    /// parses the header, and returns the size of the header + the parsed data.
    pub async fn read_metadata(buffer: &Blob) -> GenericResult<(usize, Metadata)> {
        let buffer_len: usize = buffer.len();
        if buffer_len < 8 {
            return Err(SafeTensorError::HeaderTooSmall.into());
        }

        let arr = buffer.get_bytes(0, 8).await?;

        let arr: [u8; 8] = [
            arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
        ];

        let n: usize = u64::from_le_bytes(arr)
            .try_into()
            .map_err(|_| SafeTensorError::HeaderTooLarge)?;
        if n > MAX_HEADER_SIZE {
            return Err(SafeTensorError::HeaderTooLarge.into());
        }

        let stop = n
            .checked_add(8)
            .ok_or(SafeTensorError::InvalidHeaderLength)?;
        if stop > buffer_len {
            return Err(SafeTensorError::InvalidHeaderLength.into());
        }

        let data = buffer.get_bytes(8, n).await?;

        let string = std::str::from_utf8(&data).map_err(|_| SafeTensorError::InvalidHeader)?;

        // Assert the string starts with {
        // NOTE: Add when we move to 0.4.0
        // if !string.starts_with('{') {
        //     return Err(SafeTensorError::InvalidHeaderStart);
        // }
        let metadata: Metadata = serde_json::from_str(string)
            .map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;
        let buffer_end = metadata.validate()?;
        if buffer_end + 8 + n != buffer_len {
            return Err(SafeTensorError::MetadataIncompleteBuffer.into());
        }
        Ok((n, metadata))
    }
    /// Given a byte-buffer representing the whole safetensor file
    /// parses it and returns the Deserialized form (No Tensor allocation).
    ///
    /// ```
    /// use safetensors::SafeTensors;
    /// use memmap2::MmapOptions;
    /// use std::fs::File;
    ///
    /// let filename = "model.safetensors";
    /// # use std::io::Write;
    /// # let serialized = b"<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    /// # File::create(filename).unwrap().write(serialized).unwrap();
    /// let file = File::open(filename).unwrap();
    /// let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    /// let tensors = SafeTensors::deserialize(&buffer).unwrap();
    /// let tensor = tensors
    ///         .tensor("test")
    ///         .unwrap();
    /// ```
    pub async fn deserialize(buffer: Blob) -> GenericResult<Self> {
        //let mut stream = buffer.get_stream()?;
        let (n, metadata) = SafeTensors::read_metadata(&buffer).await?;
        Ok(Self {
            metadata,
            data: buffer,
            data_offset: n + 8,
        })
    }

    /// Allow the user to iterate over tensors within the SafeTensors.
    /// The tensors returned are merely views and the data is not owned by this
    /// structure.
    pub async fn tensors(&self) -> GenericResult<Vec<(String, TensorView)>> {
        let mut tensors = Vec::with_capacity(self.metadata.index_map.len());
        for (name, &index) in &self.metadata.index_map {
            let info = &self.metadata.tensors[index];
            let tensorview = TensorView {
                dtype: info.dtype,
                shape: info.shape.clone(),
                data: self
                    .data
                    .get_bytes(
                        self.data_offset + info.data_offsets.0,
                        info.data_offsets.1 - info.data_offsets.0,
                    )
                    .await?,
            };
            tensors.push((name.to_string(), tensorview));
        }
        Ok(tensors)
    }

    /// Allow the user to get a specific tensor within the SafeTensors.
    /// The tensor returned is merely a view and the data is not owned by this
    /// structure.
    pub async fn tensor(&self, tensor_name: &str) -> GenericResult<TensorView> {
        if let Some(index) = &self.metadata.index_map.get(tensor_name) {
            if let Some(info) = &self.metadata.tensors.get(**index) {
                Ok(TensorView {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    data: self
                        .data
                        .get_bytes(
                            self.data_offset + info.data_offsets.0,
                            info.data_offsets.1 - info.data_offsets.0,
                        )
                        .await?,
                })
            } else {
                Err(SafeTensorError::TensorNotFound(tensor_name.to_string()).into())
            }
        } else {
            Err(SafeTensorError::TensorNotFound(tensor_name.to_string()).into())
        }
    }

    /// Return the names of the tensors within the SafeTensors.
    /// These are used as keys to access to the actual tensors, that can be
    /// retrieved using the tensor method.
    pub fn names(&self) -> Vec<&'_ String> {
        self.metadata.index_map.keys().collect()
    }

    /// Return how many tensors are currently stored within the SafeTensors.
    #[inline]
    pub fn len(&self) -> usize {
        self.metadata.tensors.len()
    }

    /// Indicate if the SafeTensors contains or not any tensor.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.metadata.tensors.is_empty()
    }
}

/// The stuct representing the header of safetensor files which allow
/// indexing into the raw byte-buffer array and how to interpret it.
#[derive(Debug, Clone)]
pub struct Metadata {
    metadata: Option<HashMap<String, String>>,
    tensors: Vec<TensorInfo>,
    index_map: HashMap<String, usize>,
}

/// Helper struct used only for serialization deserialization
#[derive(Serialize, Deserialize)]
struct HashMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "__metadata__")]
    metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

impl<'de> Deserialize<'de> for Metadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let hashdata: HashMetadata = HashMetadata::deserialize(deserializer)?;
        let (metadata, tensors) = (hashdata.metadata, hashdata.tensors);
        let mut tensors: Vec<_> = tensors.into_iter().collect();
        // We need to sort by offsets
        // Previous versions might have a different ordering
        // Than we expect (Not aligned ordered, but purely name ordered,
        // or actually any order).
        tensors.sort_by(|(_, left), (_, right)| left.data_offsets.cmp(&right.data_offsets));
        Metadata::new(metadata, tensors).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Metadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut names = vec![""; self.index_map.len()];
        for (name, index) in &self.index_map {
            names[*index] = name;
        }

        let tensors: Vec<_> = names.iter().zip(self.tensors.iter()).collect();
        let mut map = serializer.serialize_map(Some(tensors.len()))?;
        if let Some(metadata) = &self.metadata {
            map.serialize_entry("__metadata__", metadata)?;
        }
        for (name, info) in tensors {
            map.serialize_entry(&name, &info)?;
        }
        map.end()
    }
}

impl Metadata {
    fn new(
        metadata: Option<HashMap<String, String>>,
        tensors: Vec<(String, TensorInfo)>,
    ) -> Result<Self, SafeTensorError> {
        let mut index_map = HashMap::with_capacity(tensors.len());

        let tensors: Vec<_> = tensors
            .into_iter()
            .enumerate()
            .map(|(index, (k, tensor))| {
                index_map.insert(k, index);
                tensor
            })
            .collect();

        let metadata = Self {
            metadata,
            tensors,
            index_map,
        };
        // metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<usize, SafeTensorError> {
        let mut start = 0;
        for (i, info) in self.tensors.iter().enumerate() {
            let (s, e) = info.data_offsets;
            if s != start || e < s {
                let tensor_name = self
                    .index_map
                    .iter()
                    .find_map(|(name, &index)| if index == i { Some(&name[..]) } else { None })
                    .unwrap_or("no_tensor");
                return Err(SafeTensorError::InvalidOffset(tensor_name.to_string()));
            }
            start = e;
            let nelements: usize = info
                .shape
                .iter()
                .cloned()
                .try_fold(1usize, usize::checked_mul)
                .ok_or(SafeTensorError::ValidationOverflow)?;
            let nbytes = nelements
                .checked_mul(info.dtype.size())
                .ok_or(SafeTensorError::ValidationOverflow)?;
            if (e - s) != nbytes {
                return Err(SafeTensorError::TensorInvalidInfo);
            }
        }
        Ok(start)
    }

    /// Gives back the tensor metadata
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        let index = self.index_map.get(name)?;
        self.tensors.get(*index)
    }

    /// Gives back the tensor metadata
    pub fn tensors(&self) -> HashMap<String, &TensorInfo> {
        self.index_map
            .iter()
            .map(|(tensor_name, index)| (tensor_name.clone(), &self.tensors[*index]))
            .collect()
    }

    /// Gives back the tensor metadata
    pub fn metadata(&self) -> &Option<HashMap<String, String>> {
        &self.metadata
    }
}

/// A view of a Tensor within the file.
/// Contains references to data within the full byte-buffer
/// And is thus a readable view of a single tensor
#[derive(Debug, PartialEq, Eq)]
pub struct TensorView {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &TensorView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        (&self.data).into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl View for TensorView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        (&self.data).into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl TensorView {
    /// Create new tensor view
    pub fn new(dtype: Dtype, shape: Vec<usize>, data: Vec<u8>) -> Result<Self, SafeTensorError> {
        let n = data.len();
        let n_elements: usize = shape.iter().product();
        if n != n_elements * dtype.size() {
            Err(SafeTensorError::InvalidTensorView(dtype, shape, n))
        } else {
            Ok(Self { dtype, shape, data })
        }
    }
    /// The current tensor dtype
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The current tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The current tensor byte-buffer
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}
