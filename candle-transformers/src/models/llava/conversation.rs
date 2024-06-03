pub enum SeparatorStyle {
    Two,
    Mpt,
}
pub struct Conversation {
    pub system: String,
    pub roles: Vec<String>,
    pub messages: Vec<(String, Option<String>)>,
    pub offset: i32,
    pub sep_style: SeparatorStyle,
    pub sep: String,
    pub sep2: Option<String>,
    pub version: String,
}

impl Conversation {
    pub fn new(
        system: &str,
        roles: &[String],
        offset: i32,
        sep_style: SeparatorStyle,
        sep: &str,
        sep2: Option<&str>,
        version: &str,
    ) -> Self {
        Conversation {
            system: system.to_string(),
            roles: roles.to_vec(),
            messages: Vec::new(),
            offset,
            sep_style,
            sep: sep.to_string(),
            sep2: sep2.map(|s| s.to_string()),
            version: version.to_string(),
        }
    }

    pub fn conv_chatml_direct() -> Self {
        Conversation::new(
            "<|im_start|>system\nAnswer the questions.",
            &[
                "<|im_start|>user\n".to_string(),
                "<|im_start|>assistant\n".to_string(),
            ],
            0,
            SeparatorStyle::Mpt,
            "<|im_end|>",
            None,
            "mpt",
        )
    }

    pub fn conv_llava_v1() -> Self {
        Conversation::new(
            "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
            &[
                "USER".to_string(),
                "ASSISTANT".to_string(),
            ],
            0,
            SeparatorStyle::Two,
            " ",
            Some("</s>"),
            "v1"
        )
    }

    pub fn append_message(&mut self, role: String, message: Option<&str>) {
        self.messages.push((role, message.map(|s| s.to_string())))
    }

    pub fn append_user_message(&mut self, message: Option<&str>) {
        self.append_message(self.roles[0].clone(), message);
    }

    pub fn append_assistant_message(&mut self, message: Option<&str>) {
        self.append_message(self.roles[1].clone(), message);
    }

    pub fn get_prompt(&self) -> String {
        match self.sep_style {
            SeparatorStyle::Mpt => {
                let mut ret = String::new();
                ret.push_str(&self.system);
                ret.push_str(&self.sep);
                for (role, message) in &self.messages {
                    ret.push_str(role);
                    if let Some(message) = message {
                        ret.push_str(message);
                    };
                    ret.push_str(&self.sep);
                }
                ret
            }
            SeparatorStyle::Two => {
                let seps = [self.sep.clone(), self.sep2.clone().unwrap()];
                let mut ret = String::new();
                ret.push_str(&self.system);
                ret.push_str(&seps[0]);
                for (i, (role, message)) in self.messages.iter().enumerate() {
                    ret.push_str(role);
                    if let Some(message) = message {
                        ret.push_str(": "); // strictly follow the python implementation, otherwise it will cause some minor difference between tokens ^_^
                        ret.push_str(message);
                        ret.push_str(&seps[i % 2]);
                    } else {
                        ret.push(':')
                    }
                }
                ret
            }
        }
    }
}
